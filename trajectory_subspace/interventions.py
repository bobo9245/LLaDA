"""Hidden-state interventions compatible with :mod:`sampling`.

Three modes are supported:

- :class:`ProjectionAblation` — projects hidden states onto the orthogonal
  complement of a learned subspace ``U``.
- :class:`Steering` — adds ``alpha * v`` to every token's hidden state at the
  target layer. ``v`` is taken as the top-1 principal component of ``U`` by
  default, but can be overridden.
- :class:`RandomSubspaceControl` — same shape as ``ProjectionAblation`` but
  with a random orthonormal basis of the same rank, for sanity-check
  baselines.

Interventions are keyed by ``(layer_idx, target_ratio)``. For diffusion
sampling, the sampler does not currently gate interventions by ratio — the
hook fires every forward. If you need ratio gating, wrap the intervention in a
small observer that toggles ``active`` based on the current state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

from .sampling import Intervention


@dataclass
class ProjectionAblation(Intervention):
    """Remove a subspace from hidden states: ``h <- h - U U^T h``.

    Args:
        basis: Orthonormal matrix of shape (H, k). Columns span the subspace
            to ablate.
        target_layers: Layer indices to intervene on. Empty ⇒ every captured
            layer.
        token_slice: Optional ``slice`` restricting the intervention to a
            subset of the token axis (e.g. response-only).
    """

    basis: torch.Tensor = None
    target_layers: Sequence[int] = ()
    token_slice: Optional[slice] = None

    def apply(
        self, layer_idx: int, hidden: torch.Tensor, context: Mapping[str, Any]
    ) -> torch.Tensor:
        if self.basis is None:
            return hidden
        U = self.basis.to(device=hidden.device, dtype=hidden.dtype)
        sl = self.token_slice or slice(context.get("prompt_length", 0), None)
        h = hidden.clone()
        resp = h[:, sl, :]
        proj = resp @ U @ U.transpose(-1, -2)
        h[:, sl, :] = resp - proj
        return h


@dataclass
class Steering(Intervention):
    """Add ``alpha * v`` to the hidden state at the target layer.

    ``v`` should be a 1-D tensor of size H.
    """

    direction: torch.Tensor = None
    alpha: float = 1.0
    target_layers: Sequence[int] = ()
    token_slice: Optional[slice] = None

    def apply(
        self, layer_idx: int, hidden: torch.Tensor, context: Mapping[str, Any]
    ) -> torch.Tensor:
        if self.direction is None:
            return hidden
        v = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        sl = self.token_slice or slice(context.get("prompt_length", 0), None)
        h = hidden.clone()
        h[:, sl, :] = h[:, sl, :] + self.alpha * v
        return h


@dataclass
class RandomSubspaceControl(Intervention):
    """Ablate a random orthonormal subspace of the same rank.

    Used as a negative control: if the "real" subspace ``U`` is meaningful,
    ablating it should hurt the target behavior more than ablating a random
    same-dimensional subspace.
    """

    hidden_size: int = 0
    k: int = 1
    seed: int = 0
    target_layers: Sequence[int] = ()
    token_slice: Optional[slice] = None

    def __post_init__(self) -> None:
        gen = torch.Generator().manual_seed(self.seed)
        raw = torch.randn(self.hidden_size, self.k, generator=gen)
        # QR => orthonormal columns.
        q, _ = torch.linalg.qr(raw, mode="reduced")
        self._basis = q

    def apply(
        self, layer_idx: int, hidden: torch.Tensor, context: Mapping[str, Any]
    ) -> torch.Tensor:
        U = self._basis.to(device=hidden.device, dtype=hidden.dtype)
        sl = self.token_slice or slice(context.get("prompt_length", 0), None)
        h = hidden.clone()
        resp = h[:, sl, :]
        proj = resp @ U @ U.transpose(-1, -2)
        h[:, sl, :] = resp - proj
        return h
