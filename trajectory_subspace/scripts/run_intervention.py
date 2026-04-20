"""Apply a subspace-based intervention to LLaDA generation.

Given a cell directory written by ``collect_activations.py`` and a labeled
manifest of prompts, this script:

1. Loads the pooled activations for the requested ``(target_ratio, layer,
   pool, actual_group)`` cell.
2. Computes the top-k principal components of the centered matrix as the
   intervention subspace ``U``.
3. Runs LLaDA generation for every prompt × mode, where modes are one of:
   ``projection_ablation``, ``steering``, ``random_subspace_control``.
4. Writes a JSONL file containing, per prompt × mode, the generated response,
   a heuristic refusal score, and any metadata needed downstream.

The intervention hook runs at every forward pass on the target layer. Ratio
gating (e.g. "only ablate at t<=0.5") is not implemented here — inject a
custom observer if you need it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from trajectory_subspace.interventions import (
    ProjectionAblation,
    RandomSubspaceControl,
    Steering,
)
from trajectory_subspace.io_utils import load_cell, read_jsonl, write_jsonl, yaml_load
from trajectory_subspace.sampling import SamplingConfig, reverse_diffusion_sample
from trajectory_subspace.scripts.label_responses import (
    classify_actual_group,
    heuristic_refusal_score,
)


def _fit_basis(cell_dir: Path, actual_group: str, k: int) -> torch.Tensor:
    data = load_cell(cell_dir)
    idx = [i for i, r in enumerate(data["meta"]) if r.get("actual_group") == actual_group]
    if len(idx) < 2:
        raise ValueError(
            f"Not enough rows with actual_group={actual_group!r} in {cell_dir}"
        )
    X = data["values"].float().numpy()[idx]
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T[:, : min(k, Vt.shape[0])]
    return torch.from_numpy(V).float()


def _build_intervention(mode: str, basis: torch.Tensor, layer: int, alpha: float):
    if mode == "projection_ablation":
        return ProjectionAblation(basis=basis, target_layers=(layer,))
    if mode == "random_subspace_control":
        return RandomSubspaceControl(
            hidden_size=basis.shape[0],
            k=basis.shape[1],
            seed=0,
            target_layers=(layer,),
        )
    if mode == "steering":
        direction = basis[:, 0]
        return Steering(direction=direction, alpha=alpha, target_layers=(layer,))
    raise ValueError(f"Unknown intervention mode: {mode}")


def _load_model(cfg: Dict[str, Any]):
    from transformers import AutoModel, AutoTokenizer

    m_cfg = cfg["model"]
    dtype = getattr(torch, m_cfg.get("dtype", "bfloat16"))
    model = (
        AutoModel.from_pretrained(
            m_cfg["path"], trust_remote_code=m_cfg.get("trust_remote_code", True),
            torch_dtype=dtype,
        )
        .to("cuda")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        m_cfg["path"], trust_remote_code=m_cfg.get("trust_remote_code", True)
    )
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cell-dir", type=Path, required=True)
    p.add_argument("--labeled-manifest", type=Path, required=True)
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--mode", required=True,
                   choices=["projection_ablation", "steering", "random_subspace_control"])
    p.add_argument("--fit-group", default="T_plus",
                   help="Which actual_group to fit the subspace on.")
    p.add_argument("--k", type=int, default=4,
                   help="Subspace rank.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Steering coefficient.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml_load(args.config)
    sampling = cfg["sampling"]

    basis = _fit_basis(args.cell_dir, args.fit_group, args.k)
    intervention = _build_intervention(args.mode, basis, args.layer, args.alpha)

    rows = list(read_jsonl(args.labeled_manifest))
    if args.limit is not None:
        rows = rows[: args.limit]

    model, tokenizer = _load_model(cfg)

    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=True, tokenize=False
        )
        enc = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        s_cfg = SamplingConfig(
            steps=sampling["steps"],
            gen_length=sampling["gen_length"],
            block_length=sampling["block_length"],
            temperature=sampling.get("temperature", 0.0),
            cfg_scale=sampling.get("cfg_scale", 0.0),
            remasking=sampling.get("remasking", "low_confidence"),
            mask_id=int(sampling.get("mask_id", 126336)),
            # No activation capture needed, but the intervention hook needs
            # to fire -> request at least the target layer.
            capture_layers=(args.layer,),
        )
        out = reverse_diffusion_sample(
            model,
            input_ids,
            s_cfg,
            attention_mask=attention_mask,
            intervention=intervention,
        )
        resp = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
        ref = heuristic_refusal_score(resp)
        actual = classify_actual_group(row.get("intended_group", "unknown"), ref)
        out_rows.append(
            {
                **row,
                "intervention_mode": args.mode,
                "intervention_layer": args.layer,
                "intervention_k": args.k,
                "intervention_alpha": args.alpha,
                "intervention_fit_group": args.fit_group,
                "post_response": resp,
                "post_refusal_score": ref,
                "post_actual_group": actual,
            }
        )

    write_jsonl(args.out_path, out_rows)
    print(f"Wrote {len(out_rows)} intervention rows to {args.out_path}")


if __name__ == "__main__":
    main()
