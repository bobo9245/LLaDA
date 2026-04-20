"""Observers fed by :func:`trajectory_subspace.sampling.reverse_diffusion_sample`.

The main concrete observer is :class:`CollectObserver`, which, for a fixed set
of response-masked-fraction targets and layer indices, captures pooled hidden
states (and optionally token-level hidden states and final logits) keyed by
``(target_ratio, layer, pooling)``.

Pooling strategies:

- ``response_mean``: mean over all response positions (masked + unmasked).
- ``last_unmasked``: hidden state at the highest-index response position that
  is already unmasked *after* the current step; falls back to the last
  response position if all are masked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .sampling import ForwardOutput, SamplingConfig, SamplingState, TrajectoryObserver


@dataclass
class _CellRecord:
    """Captured activations / metadata for a single ``(target, layer, pool)`` cell."""

    pooled: torch.Tensor                # (B, H)
    token_level: Optional[torch.Tensor] = None  # (B, G, H) — response slice
    actual_ratio: torch.Tensor = None   # (B,)
    global_step: int = -1
    block_index: int = -1
    logits_response: Optional[torch.Tensor] = None  # (B, G, V) if kept


@dataclass
class _CaptureState:
    """Per-sample state tracking which targets have fired."""

    # Target index -> True if already captured.
    fired: List[bool]
    # Previous response-masked fraction (so we can detect crossing from above).
    prev_ratio: float = 1.0


class CollectObserver(TrajectoryObserver):
    """Capture pooled hidden states at crossing points.

    A target ratio *t* is "hit" the first step at which the per-sample masked
    fraction falls at or below *t* (from above). This mirrors the notion of a
    diffusion timestep more faithfully than "step number" and is invariant to
    changes in ``steps`` or ``block_length``.
    """

    def __init__(
        self,
        target_ratios: Sequence[float],
        layers: Sequence[int],
        poolings: Sequence[str] = ("response_mean",),
        keep_token_level: bool = False,
        keep_response_logits: bool = False,
    ) -> None:
        self.target_ratios = tuple(float(r) for r in target_ratios)
        self.layers = tuple(int(l) for l in layers)
        self.poolings = tuple(poolings)
        self.keep_token_level = keep_token_level
        self.keep_response_logits = keep_response_logits
        # Key: (target_idx, layer, pool) -> record
        self.records: Dict[Tuple[int, int, str], _CellRecord] = {}
        self._state: List[_CaptureState] = []
        self.wants_hidden_states = True

    # -- TrajectoryObserver API ---------------------------------------------

    def on_sampling_start(self, config: SamplingConfig, prompt: torch.Tensor) -> None:
        B = prompt.shape[0]
        self._state = [
            _CaptureState(fired=[False] * len(self.target_ratios), prev_ratio=1.0)
            for _ in range(B)
        ]

    def on_step_end(self, state: SamplingState, forward: ForwardOutput) -> None:
        B = state.x.shape[0]
        ratios = state.response_masked_fraction.detach().cpu().tolist()

        for t_idx, target in enumerate(self.target_ratios):
            fire_mask = [
                (not self._state[b].fired[t_idx])
                and (self._state[b].prev_ratio > target)
                and (ratios[b] <= target)
                for b in range(B)
            ]
            if not any(fire_mask):
                continue
            # Fire every layer for any samples that crossed.
            for layer in self.layers:
                hidden = self._hidden_for_layer(forward, layer)
                if hidden is None:
                    continue
                resp_hidden = hidden[:, state.prompt_length :, :]
                resp_tokens = state.x[:, state.prompt_length :]
                mask_resp = resp_tokens == state.mask_id
                for pool in self.poolings:
                    pooled = self._pool(resp_hidden, mask_resp, pool)
                    self._merge(
                        key=(t_idx, layer, pool),
                        pooled=pooled,
                        fire_mask=fire_mask,
                        ratios=ratios,
                        state=state,
                        resp_hidden=resp_hidden if self.keep_token_level else None,
                        logits=(
                            forward.logits[:, state.prompt_length :]
                            if self.keep_response_logits
                            else None
                        ),
                    )
            for b, fired in enumerate(fire_mask):
                if fired:
                    self._state[b].fired[t_idx] = True

        for b in range(B):
            self._state[b].prev_ratio = ratios[b]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _hidden_for_layer(
        forward: ForwardOutput, layer: int
    ) -> Optional[torch.Tensor]:
        if forward.hidden_states is None:
            return None
        if layer < 0 or layer >= len(forward.hidden_states):
            return None
        return forward.hidden_states[layer]

    @staticmethod
    def _pool(
        resp_hidden: torch.Tensor, mask_resp: torch.Tensor, pool: str
    ) -> torch.Tensor:
        if pool == "response_mean":
            return resp_hidden.mean(dim=1)
        if pool == "last_unmasked":
            # Highest-index response position that is *not* masked after step.
            B, G, H = resp_hidden.shape
            not_masked = (~mask_resp).float()
            idx_weight = not_masked * torch.arange(
                G, device=resp_hidden.device, dtype=resp_hidden.dtype
            ).unsqueeze(0)
            # Where no position is unmasked, fall back to last position.
            any_unmasked = not_masked.sum(dim=1) > 0
            last_idx = torch.where(
                any_unmasked,
                idx_weight.argmax(dim=1),
                torch.full((B,), G - 1, device=resp_hidden.device, dtype=torch.long),
            )
            return resp_hidden.gather(
                1, last_idx.view(B, 1, 1).expand(B, 1, H)
            ).squeeze(1)
        raise ValueError(f"Unknown pooling: {pool}")

    def _merge(
        self,
        key: Tuple[int, int, str],
        pooled: torch.Tensor,
        fire_mask: List[bool],
        ratios: List[float],
        state: SamplingState,
        resp_hidden: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
    ) -> None:
        fire_idx = [b for b, f in enumerate(fire_mask) if f]
        fire_idx_t = torch.tensor(fire_idx, device=pooled.device, dtype=torch.long)
        pooled_slice = pooled.index_select(0, fire_idx_t).detach().cpu()
        actual = torch.tensor(
            [ratios[b] for b in fire_idx], dtype=torch.float32
        )
        token_slice = (
            resp_hidden.index_select(0, fire_idx_t).detach().cpu()
            if resp_hidden is not None
            else None
        )
        logits_slice = (
            logits.index_select(0, fire_idx_t).detach().cpu()
            if logits is not None
            else None
        )
        existing = self.records.get(key)
        if existing is None:
            self.records[key] = _CellRecord(
                pooled=pooled_slice,
                token_level=token_slice,
                actual_ratio=actual,
                global_step=state.global_step,
                block_index=state.block_index,
                logits_response=logits_slice,
            )
            return
        existing.pooled = torch.cat([existing.pooled, pooled_slice], dim=0)
        existing.actual_ratio = torch.cat([existing.actual_ratio, actual], dim=0)
        if token_slice is not None and existing.token_level is not None:
            existing.token_level = torch.cat([existing.token_level, token_slice], dim=0)
        if logits_slice is not None and existing.logits_response is not None:
            existing.logits_response = torch.cat(
                [existing.logits_response, logits_slice], dim=0
            )
