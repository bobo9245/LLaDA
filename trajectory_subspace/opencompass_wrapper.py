"""OpenCompass-compatible LLaDA model that applies an intervention at forward.

Usage — point an OpenCompass config at this class instead of ``LLaDAModel``.
Example::

    from trajectory_subspace.opencompass_wrapper import IntervenedLLaDAModel

    models = [
        dict(
            type=IntervenedLLaDAModel,
            abbr="llada-8b-instruct-ablated",
            path="/mnt/oujingyang/assets/model/LLaDA",
            max_out_len=1024,
            batch_size=1,
            run_cfg=dict(num_gpus=1),
            intervention=dict(
                mode="projection_ablation",
                basis_path="out/activations/pooled/all/t=0.50_l=14_p=response_mean/",
                fit_group="T_plus",
                k=4,
                layer=14,
            ),
        )
    ]

``basis_path`` points at a cell directory produced by
``collect_activations.py``. The wrapper fits the basis once at init and
reuses it for every generation call.

``LLaDAModel`` assumes the LLaDA repo root is on ``sys.path``; we use the
same bootstrap as ``opencompass/opencompass/models/dllm.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Make sure the LLaDA repo root is on sys.path so ``from generate import
# generate`` (used by the upstream ``LLaDAModel``) still works.
_LLADA_ROOT = Path(__file__).resolve().parents[1]
if str(_LLADA_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLADA_ROOT))

try:
    from opencompass.models.dllm import LLaDAModel  # type: ignore
    from opencompass.registry import MODELS  # type: ignore
    _OPENCOMPASS_AVAILABLE = True
except Exception:  # pragma: no cover - OpenCompass optional.
    LLaDAModel = object  # type: ignore
    MODELS = None  # type: ignore
    _OPENCOMPASS_AVAILABLE = False

from trajectory_subspace.interventions import (
    ProjectionAblation,
    RandomSubspaceControl,
    Steering,
)
from trajectory_subspace.io_utils import load_cell
from trajectory_subspace.sampling import SamplingConfig, reverse_diffusion_sample


def _fit_basis(basis_path: str, fit_group: str, k: int) -> torch.Tensor:
    data = load_cell(basis_path)
    idx = [i for i, r in enumerate(data["meta"]) if r.get("actual_group") == fit_group]
    if len(idx) < 2:
        raise ValueError(
            f"Not enough rows with actual_group={fit_group!r} in {basis_path}"
        )
    X = data["values"].float().numpy()[idx]
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T[:, : min(k, Vt.shape[0])]
    return torch.from_numpy(V).float()


def _build_intervention(intervention_cfg: Dict[str, Any]):
    mode = intervention_cfg["mode"]
    layer = int(intervention_cfg["layer"])
    if mode in {"projection_ablation", "steering"}:
        basis = _fit_basis(
            intervention_cfg["basis_path"],
            intervention_cfg.get("fit_group", "T_plus"),
            int(intervention_cfg.get("k", 4)),
        )
        if mode == "projection_ablation":
            return ProjectionAblation(basis=basis, target_layers=(layer,))
        return Steering(
            direction=basis[:, 0],
            alpha=float(intervention_cfg.get("alpha", 1.0)),
            target_layers=(layer,),
        )
    if mode == "random_subspace_control":
        return RandomSubspaceControl(
            hidden_size=int(intervention_cfg["hidden_size"]),
            k=int(intervention_cfg.get("k", 4)),
            seed=int(intervention_cfg.get("seed", 0)),
            target_layers=(layer,),
        )
    raise ValueError(f"Unknown intervention mode: {mode}")


def _register(cls):
    if _OPENCOMPASS_AVAILABLE and MODELS is not None:
        return MODELS.register_module()(cls)
    return cls


@_register
class IntervenedLLaDAModel(LLaDAModel):  # type: ignore[misc]
    """LLaDAModel with a hidden-state intervention applied during forward.

    Takes the same kwargs as :class:`opencompass.models.LLaDAModel`, plus an
    ``intervention`` dict (see module docstring).
    """

    def __init__(self, intervention: Optional[Dict[str, Any]] = None, **kwargs):
        if not _OPENCOMPASS_AVAILABLE:
            raise RuntimeError(
                "OpenCompass is not importable from this environment. "
                "Install it or run interventions via scripts/run_intervention.py."
            )
        super().__init__(**kwargs)
        self._intervention = (
            _build_intervention(intervention) if intervention is not None else None
        )
        self._intervention_layer = (
            int(intervention["layer"]) if intervention is not None else None
        )

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:  # type: ignore[override]
        if self._intervention is None:
            return super().generate(inputs, max_out_len)

        from opencompass.models.dllm import _convert_chat_messages  # type: ignore

        messages = _convert_chat_messages(inputs)
        rendered = [
            self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages
        ]
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer.batch_encode_plus(
            rendered, padding=True, return_tensors="pt"
        )
        prompt = enc["input_ids"].to(self.model.device)
        attn = enc.get("attention_mask")
        attn = attn.to(self.model.device) if attn is not None else None

        s_cfg = SamplingConfig(
            steps=self.gen_steps,
            gen_length=self.gen_length,
            block_length=self.gen_blocksize,
            temperature=self.temperature,
            cfg_scale=self.cfg,
            remasking=self.remasking,
            mask_id=self.mask_id,
            logits_eos_inf=getattr(self, "diff_logits_eos_inf", False),
            confidence_eos_eot_inf=getattr(self, "diff_confidence_eos_eot_inf", False),
            capture_layers=(self._intervention_layer,)
            if self._intervention_layer is not None
            else (),
        )
        out = reverse_diffusion_sample(
            self.model,
            prompt,
            s_cfg,
            attention_mask=attn,
            intervention=self._intervention,
        )
        return [
            self.tokenizer.decode(out[i, -self.gen_length :], skip_special_tokens=True)
            for i in range(prompt.shape[0])
        ]
