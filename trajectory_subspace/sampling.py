"""Shared reverse-diffusion sampling core for LLaDA.

The original sampling loop lived in ``generate.py``. It is extracted here so
that (a) activation-collection experiments and (b) interventions can plug into
the same loop without maintaining a second copy. ``generate.generate`` is kept
as a thin wrapper around :func:`reverse_diffusion_sample` so every caller
(``chat.py``, ``eval_llada.py``, the OpenCompass wrapper) keeps its existing
signature and byte-for-byte behavior when no observer / intervention is
supplied.

Design:

* ``SamplingConfig`` carries the knobs that used to be ``generate()`` kwargs.
* ``TrajectoryObserver.on_step_end`` gets a lightweight dataclass of state at
  every step, without copying the full hidden-state stack unless the observer
  asks for it (``wants_hidden_states``).
* Hidden states are captured natively through ``output_hidden_states=True``
  when the underlying model accepts it, and fall back automatically to forward
  hooks registered on the transformer blocks. Layer indices are the contiguous
  index over the discovered block list, so downstream code always sees
  ``0..L-1`` regardless of how the model exposes them internally.
* ``Intervention.apply`` is invoked for every requested layer on every forward
  and edits the hidden state in place (via hook).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Knobs for reverse-diffusion sampling.

    Fields mirror the original ``generate()`` signature plus a small number of
    capture-related knobs. Defaults match the values hard-coded in
    ``generate.main`` so existing eval scripts produce identical output.
    """

    steps: int = 128
    gen_length: int = 128
    block_length: int = 128
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False

    # Layers to expose to observers / interventions. Empty list means no
    # hidden-state capture at all (the fast path used by plain generation).
    capture_layers: Sequence[int] = field(default_factory=tuple)

    # If ``True`` the sampler will populate ``ForwardOutput.hidden_states`` for
    # every forward pass. Observers that only need a subset should leave this
    # ``False`` and request layers via ``capture_layers``.
    want_full_hidden_states: bool = False


@dataclass
class SamplingState:
    """Snapshot handed to an observer after each reverse-diffusion step."""

    step: int                  # 0-indexed step within the current block
    global_step: int           # 0-indexed step across all blocks
    block_index: int
    num_blocks: int
    steps_per_block: int
    prompt_length: int
    gen_length: int
    mask_id: int
    # Full sequence tensor (prompt + response) after this step, shape (B, L)
    x: torch.Tensor
    # Indices of positions that were unmasked *during this step*, as a bool
    # tensor of shape (B, L).
    transfer_index: torch.Tensor
    # Mask of positions that were still masked *before* this step.
    mask_index_before: torch.Tensor
    # Per-position confidence tensor produced by the remasking strategy.
    confidence: torch.Tensor
    # Fraction of response positions still masked *after* this step, shape (B,).
    response_masked_fraction: torch.Tensor


@dataclass
class ForwardOutput:
    """Outputs from a single model forward pass, possibly with hidden states."""

    logits: torch.Tensor
    # hidden_states[layer_idx] is (B, L, H). Only populated for layers the
    # caller asked for; ``None`` for layers that were not requested.
    hidden_states: Optional[List[Optional[torch.Tensor]]] = None


# ---------------------------------------------------------------------------
# Observer / Intervention protocols
# ---------------------------------------------------------------------------


class TrajectoryObserver:
    """Base class for step-level sampling observers.

    Subclasses override :meth:`on_step_end`. The sampler also consults
    :attr:`wants_hidden_states` to decide whether to run the model with
    ``output_hidden_states=True`` / install fallback hooks.
    """

    wants_hidden_states: bool = False

    def on_sampling_start(self, config: SamplingConfig, prompt: torch.Tensor) -> None:
        pass

    def on_step_end(self, state: SamplingState, forward: ForwardOutput) -> None:
        pass

    def on_sampling_end(self, final_x: torch.Tensor) -> None:
        pass


class NullObserver(TrajectoryObserver):
    """No-op observer; used on the fast path to preserve original behavior."""

    wants_hidden_states = False


class Intervention:
    """Base class for hidden-state interventions.

    Subclasses implement :meth:`apply`. The sampler registers a single forward
    hook per requested layer and routes every hidden-state tensor through it.
    """

    # Layers this intervention targets. Empty => applies to every captured
    # layer. Indices refer to the contiguous transformer-block index (0..L-1).
    target_layers: Sequence[int] = ()

    def apply(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        context: Mapping[str, Any],
    ) -> torch.Tensor:
        """Return a (possibly modified) hidden-state tensor.

        ``context`` currently carries ``{"prompt_length": int}``; more fields
        may be added as we learn which are useful. Always return a tensor of
        the same shape / dtype / device as ``hidden``.
        """
        return hidden


class NullIntervention(Intervention):
    def apply(self, layer_idx, hidden, context):  # pragma: no cover - trivial
        return hidden


# ---------------------------------------------------------------------------
# Helpers copied verbatim from ``generate.py`` — we keep them here too so the
# sampling module is self-contained. ``generate.py`` re-exports them.
# ---------------------------------------------------------------------------


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


# ---------------------------------------------------------------------------
# Hidden-state capture
# ---------------------------------------------------------------------------


def _discover_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Find the contiguous list of transformer blocks.

    LLaDA uses ``trust_remote_code=True`` models whose block list may live at a
    few different paths (``model.transformer.blocks``,
    ``model.model.transformer.blocks``, ``model.layers``, ...). We look for the
    longest ``ModuleList`` under the module and assume those are the blocks.
    Good enough for the handful of architectures we care about; callers can
    also inject a discovery override by setting ``model._trajectory_blocks``.
    """
    override = getattr(model, "_trajectory_blocks", None)
    if override is not None:
        return list(override)

    best: List[torch.nn.Module] = []
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > len(best):
            # Heuristic: blocks are modules whose children include attention +
            # feed-forward subcomponents. We do not verify this strictly but
            # require at least 2 submodules per entry to skip embedding lists.
            if len(module) >= 2 and all(
                sum(1 for _ in child.children()) >= 2 for child in module
            ):
                best = list(module)
    if not best:
        raise RuntimeError(
            "Could not auto-discover transformer blocks. Set "
            "``model._trajectory_blocks`` to a list of block modules."
        )
    return best


@contextlib.contextmanager
def _capture_hidden_states(
    model: torch.nn.Module,
    layers: Sequence[int],
    intervention: Optional[Intervention],
    context: Mapping[str, Any],
):
    """Install forward hooks to capture (and optionally edit) hidden states.

    Yields a ``list`` whose length matches the contiguous block count. After
    each forward pass, entries for requested ``layers`` hold ``(B, L, H)``
    tensors; the rest are ``None``. The list is cleared before each forward by
    whoever owns it.
    """
    if not layers and intervention is None:
        yield None
        return

    blocks = _discover_transformer_blocks(model)
    storage: List[Optional[torch.Tensor]] = [None] * len(blocks)
    layer_set = set(int(i) for i in layers) if layers else set()

    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            # Transformer blocks may return a tensor, a tuple, or a dataclass.
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            if intervention is not None and (
                not intervention.target_layers
                or idx in set(int(i) for i in intervention.target_layers)
            ):
                hidden = intervention.apply(idx, hidden, context)
                if isinstance(output, tuple):
                    output = (hidden,) + output[1:]
                else:
                    output = hidden
            if idx in layer_set:
                # Detach + move to CPU later if the observer wants to keep it;
                # for now we hand the live tensor to the observer so it can
                # choose. Keep it on-device to avoid cost on the fast path.
                storage[idx] = hidden.detach()
            return output

        return hook

    for i, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(i)))

    try:
        yield storage
    finally:
        for h in handles:
            h.remove()


def _try_native_hidden_states(
    model: torch.nn.Module,
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    want_hidden: bool,
) -> tuple[torch.Tensor, Optional[Sequence[torch.Tensor]]]:
    """Forward pass that asks the model for hidden_states, if it will cooperate.

    Returns (logits, hidden_tuple_or_None). If the model does not accept
    ``output_hidden_states``, we fall back to plain forward and return
    ``None`` for the hidden states; the caller then relies on hook capture.
    """
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if want_hidden:
        try:
            out = model(x, output_hidden_states=True, **kwargs)
            hidden = getattr(out, "hidden_states", None)
            return out.logits, hidden
        except TypeError:
            pass
    out = model(x, **kwargs)
    return out.logits, None


# ---------------------------------------------------------------------------
# The reusable sampling loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def reverse_diffusion_sample(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    config: SamplingConfig,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    observer: Optional[TrajectoryObserver] = None,
    intervention: Optional[Intervention] = None,
) -> torch.Tensor:
    """Run LLaDA's reverse-diffusion sampler with optional hooks.

    When ``observer`` is ``None`` (or a :class:`NullObserver`) and
    ``intervention`` is ``None``, this function produces output bit-identical
    to the original ``generate.generate`` at the same inputs.
    """
    observer = observer or NullObserver()
    cfg = config

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + cfg.gen_length),
        cfg.mask_id,
        dtype=torch.long,
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], cfg.gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != cfg.mask_id

    assert cfg.gen_length % cfg.block_length == 0, (
        "gen_length must be divisible by block_length"
    )
    num_blocks = cfg.gen_length // cfg.block_length
    assert cfg.steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = cfg.steps // num_blocks

    observer.on_sampling_start(cfg, prompt)

    want_hidden = bool(cfg.capture_layers) or getattr(
        observer, "wants_hidden_states", False
    ) or cfg.want_full_hidden_states
    need_hooks = want_hidden or (intervention is not None)
    hook_layers: Sequence[int] = (
        cfg.capture_layers if cfg.want_full_hidden_states or cfg.capture_layers else ()
    )
    context = {"prompt_length": int(prompt.shape[1])}

    with _capture_hidden_states(
        model, hook_layers, intervention, context
    ) if need_hooks else contextlib.nullcontext() as hook_storage:
        global_step = 0
        for num_block in range(num_blocks):
            block_mask_index = (
                x[
                    :,
                    prompt.shape[1] + num_block * cfg.block_length : prompt.shape[1]
                    + (num_block + 1) * cfg.block_length,
                ]
                == cfg.mask_id
            )
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )
            for i in range(steps_per_block):
                mask_index = x == cfg.mask_id
                if cfg.cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = cfg.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    attn_ = (
                        torch.cat([attention_mask, attention_mask], dim=0)
                        if attention_mask is not None
                        else None
                    )
                    logits, hidden = _try_native_hidden_states(
                        model, x_, attn_, want_hidden
                    )
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits, hidden = _try_native_hidden_states(
                        model, x, attention_mask, want_hidden
                    )

                if cfg.logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                logits_with_noise = add_gumbel_noise(logits, temperature=cfg.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if cfg.confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

                if cfg.remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif cfg.remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )
                else:
                    raise NotImplementedError(cfg.remasking)

                x0_p[
                    :, prompt.shape[1] + (num_block + 1) * cfg.block_length :
                ] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

                # Compose the hidden-state list visible to the observer.
                hidden_list: Optional[List[Optional[torch.Tensor]]] = None
                if want_hidden:
                    if hidden is not None:
                        # Native path. ``hidden`` has len = num_blocks + 1
                        # (embeddings + one per block). We align to the block
                        # indices: block i's output is hidden[i + 1].
                        layer_list: List[Optional[torch.Tensor]] = [None] * (
                            len(hidden) - 1
                        )
                        requested = (
                            set(int(l) for l in cfg.capture_layers)
                            if cfg.capture_layers
                            else set(range(len(hidden) - 1))
                        )
                        for idx in requested:
                            if 0 <= idx < len(hidden) - 1:
                                layer_list[idx] = hidden[idx + 1].detach()
                        hidden_list = layer_list
                    elif hook_storage is not None:
                        hidden_list = list(hook_storage)
                        # Reset for next forward
                        for k in range(len(hook_storage)):
                            hook_storage[k] = None

                # Response-masked fraction *after* this step.
                resp_slice = x[:, prompt.shape[1] :]
                resp_masked = (resp_slice == cfg.mask_id).float().mean(dim=1)

                state = SamplingState(
                    step=i,
                    global_step=global_step,
                    block_index=num_block,
                    num_blocks=num_blocks,
                    steps_per_block=steps_per_block,
                    prompt_length=int(prompt.shape[1]),
                    gen_length=cfg.gen_length,
                    mask_id=cfg.mask_id,
                    x=x,
                    transfer_index=transfer_index,
                    mask_index_before=mask_index,
                    confidence=confidence,
                    response_masked_fraction=resp_masked,
                )
                forward = ForwardOutput(logits=logits, hidden_states=hidden_list)
                observer.on_step_end(state, forward)
                global_step += 1

    observer.on_sampling_end(x)
    return x


# ---------------------------------------------------------------------------
# Backwards-compatible wrapper matching ``generate.generate`` semantics.
# ---------------------------------------------------------------------------


def sample_like_generate(
    model,
    prompt,
    attention_mask=None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    observer: Optional[TrajectoryObserver] = None,
    intervention: Optional[Intervention] = None,
) -> torch.Tensor:
    """Matches the positional/keyword contract of the old ``generate.generate``."""
    cfg = SamplingConfig(
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
    )
    return reverse_diffusion_sample(
        model,
        prompt,
        cfg,
        attention_mask=attention_mask,
        observer=observer,
        intervention=intervention,
    )
