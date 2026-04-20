"""Compute subspace analytics on pooled activations.

Per ``(target_ratio, layer, pooling)`` cell, compute:

- Singular spectrum of the centered pooled matrix.
- Effective rank = exp(H(p)), p = s^2 / sum(s^2).
- k90 / k95 = smallest k with cumulative explained variance >= threshold.
- Participation ratio = (sum s^2)^2 / sum(s^4).
- Held-out reconstruction error at each k.
- Principal angles between class bases (T+, T-, HN_adjacent, C).
- Principal angles across adjacent timesteps (timestep evolution).
- Principal angles vs null baselines (label_shuffle, length_match,
  topic_match).

Outputs are written to ``<out_root>/analysis/<group>/<cell>.json`` and a
combined ``summary.jsonl`` with one row per cell.

The computations here are deliberately pure numpy / torch. They do not load
models, so this script is light enough to run on a CPU box.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from trajectory_subspace.io_utils import load_cell, write_jsonl, yaml_load


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def svd_spectrum(X: np.ndarray) -> np.ndarray:
    """Return singular values of a centered (N, H) matrix."""
    Xc = _center(X)
    # Use torch.linalg.svd for parity with PyTorch-heavy callers; numpy would
    # also work.
    s = torch.linalg.svdvals(torch.from_numpy(Xc)).numpy()
    return s


def effective_rank(singular_values: np.ndarray) -> float:
    s2 = singular_values ** 2
    total = s2.sum()
    if total <= 0:
        return 0.0
    p = s2 / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(singular_values: np.ndarray) -> float:
    s2 = singular_values ** 2
    num = s2.sum() ** 2
    den = (s2 ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def k_for_cumulative(singular_values: np.ndarray, threshold: float) -> int:
    s2 = singular_values ** 2
    if s2.sum() == 0:
        return 0
    cum = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cum, threshold) + 1)


def principal_angles(
    basis_a: np.ndarray, basis_b: np.ndarray
) -> np.ndarray:
    """Principal angles (radians, ascending) between column spaces of ``A/B``.

    Both arguments should have orthonormal columns; otherwise we orthonormalize
    via QR first.
    """
    Qa, _ = np.linalg.qr(basis_a)
    Qb, _ = np.linalg.qr(basis_b)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def top_k_basis(X: np.ndarray, k: int) -> np.ndarray:
    Xc = _center(X)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    return V[:, : min(k, V.shape[1])]


def held_out_reconstruction(
    train: np.ndarray, test: np.ndarray, ks: Sequence[int]
) -> Dict[int, float]:
    """Squared-error reconstruction of ``test`` onto ``train``'s top-k basis.

    Returned metric is ``1 - ||test - proj||_F^2 / ||test||_F^2`` — the
    fraction of held-out Frobenius energy captured by the training subspace.
    """
    if train.shape[0] < 2 or test.shape[0] < 1:
        return {int(k): float("nan") for k in ks}
    mu = train.mean(axis=0, keepdims=True)
    Xc = train - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    test_c = test - mu
    test_norm_sq = float(np.linalg.norm(test_c, "fro") ** 2)
    if test_norm_sq == 0:
        return {int(k): float("nan") for k in ks}
    out: Dict[int, float] = {}
    for k in ks:
        Vk = V[:, : min(int(k), V.shape[1])]
        proj = test_c @ Vk @ Vk.T
        err = float(np.linalg.norm(test_c - proj, "fro") ** 2)
        out[int(k)] = 1.0 - err / test_norm_sq
    return out


# ---------------------------------------------------------------------------
# Cell walking
# ---------------------------------------------------------------------------


def _iter_cells(pooled_root: Path):
    """Yield (cell_dir, target, layer, pool) for every cell directory."""
    for cell_dir in sorted(pooled_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        name = cell_dir.name
        # Format: t=<f>_l=<ii>_p=<pool>
        try:
            parts = dict(
                seg.split("=", 1) for seg in name.split("_") if "=" in seg
            )
        except Exception:
            continue
        yield cell_dir, float(parts["t"]), int(parts["l"]), parts["p"]


def _group_values_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[str(r.get(key))].append(i)
    return dict(groups)


def _split_train_test(indices: List[int], held_out: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * held_out)))
    return shuffled[n_test:], shuffled[:n_test]


def _label_shuffle(labels: List[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------------------
# Cell analysis
# ---------------------------------------------------------------------------


def analyze_cell(
    cell: Dict[str, Any],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    X = cell["values"].float().numpy()
    meta = cell["meta"]
    N, H = X.shape

    thresholds = list(cfg.get("variance_thresholds", [0.9, 0.95]))
    held_out = float(cfg.get("held_out_ratio", 0.2))
    analysis_seeds = list(cfg.get("seeds", [0]))
    angle_k = int(cfg.get("principal_angle_max_k", 32))

    result: Dict[str, Any] = {
        "n_rows": N,
        "hidden_size": H,
    }

    # Global spectrum.
    s_all = svd_spectrum(X)
    result["effective_rank"] = effective_rank(s_all)
    result["participation_ratio"] = participation_ratio(s_all)
    result["singular_values_head"] = s_all[: min(16, len(s_all))].tolist()
    for th in thresholds:
        result[f"k{int(th * 100)}"] = k_for_cumulative(s_all, th)

    # Per-class spectrum + held-out reconstruction.
    groups = _group_values_by(meta, "actual_group")
    per_class: Dict[str, Any] = {}
    for cls, idx in groups.items():
        if len(idx) < 4:
            continue
        Xc = X[idx]
        s_c = svd_spectrum(Xc)
        per_class_entry: Dict[str, Any] = {
            "n": len(idx),
            "effective_rank": effective_rank(s_c),
            "participation_ratio": participation_ratio(s_c),
        }
        for th in thresholds:
            per_class_entry[f"k{int(th * 100)}"] = k_for_cumulative(s_c, th)

        # Held-out reconstruction averaged over analysis seeds.
        recons: Dict[int, List[float]] = defaultdict(list)
        for sd in analysis_seeds:
            train_i, test_i = _split_train_test(idx, held_out, sd)
            ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= len(train_i) and k <= H]
            if not ks:
                continue
            cap = held_out_reconstruction(X[train_i], X[test_i], ks)
            for k, v in cap.items():
                recons[k].append(v)
        per_class_entry["heldout_reconstruction"] = {
            k: float(np.nanmean(vs)) for k, vs in recons.items()
        }
        per_class[cls] = per_class_entry
    result["per_class"] = per_class

    # Principal angles between class subspaces (top-k bases).
    angles: Dict[str, List[float]] = {}
    class_bases = {}
    for cls, idx in groups.items():
        if len(idx) < 2:
            continue
        k_eff = min(angle_k, len(idx) - 1, H)
        if k_eff <= 0:
            continue
        class_bases[cls] = top_k_basis(X[idx], k_eff)
    class_list = sorted(class_bases)
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            a, b = class_list[i], class_list[j]
            theta = principal_angles(class_bases[a], class_bases[b])
            angles[f"{a}__vs__{b}"] = theta.tolist()
    result["principal_angles"] = angles

    # Null baseline: label-shuffle control for T+ vs T- angles.
    if "T_plus" in groups and "T_minus" in groups and angle_k > 0:
        joint = groups["T_plus"] + groups["T_minus"]
        labels = (
            ["T_plus"] * len(groups["T_plus"]) + ["T_minus"] * len(groups["T_minus"])
        )
        null_angles: List[List[float]] = []
        for sd in analysis_seeds:
            shuf = _label_shuffle(labels, sd)
            tp_i = [joint[i] for i, l in enumerate(shuf) if l == "T_plus"]
            tm_i = [joint[i] for i, l in enumerate(shuf) if l == "T_minus"]
            if len(tp_i) < 2 or len(tm_i) < 2:
                continue
            k_eff = min(angle_k, len(tp_i) - 1, len(tm_i) - 1, H)
            bp = top_k_basis(X[tp_i], k_eff)
            bm = top_k_basis(X[tm_i], k_eff)
            null_angles.append(principal_angles(bp, bm).tolist())
        result["null_label_shuffle"] = null_angles

    return result


# ---------------------------------------------------------------------------
# Timestep evolution analysis (D3)
# ---------------------------------------------------------------------------


def timestep_evolution(
    pooled_root: Path,
    layer: int,
    pool: str,
    group: str,
    angle_k: int,
) -> Dict[str, Any]:
    # Collect (target, cell) for this (layer, pool) pair.
    cells: List[Tuple[float, Path]] = []
    for cell_dir, target, l, p in _iter_cells(pooled_root):
        if l == layer and p == pool:
            cells.append((target, cell_dir))
    cells.sort(key=lambda x: -x[0])  # high mask-fraction first.

    bases: List[Tuple[float, np.ndarray]] = []
    for target, cell_dir in cells:
        data = load_cell(cell_dir)
        idx = [i for i, r in enumerate(data["meta"]) if r.get("actual_group") == group]
        if len(idx) < 4:
            continue
        X = data["values"].float().numpy()[idx]
        k_eff = min(angle_k, len(idx) - 1, X.shape[1])
        if k_eff <= 0:
            continue
        bases.append((target, top_k_basis(X, k_eff)))

    out: Dict[str, Any] = {"pairs": []}
    for i in range(len(bases) - 1):
        t_a, ba = bases[i]
        t_b, bb = bases[i + 1]
        out["pairs"].append(
            {
                "from": t_a,
                "to": t_b,
                "angles": principal_angles(ba, bb).tolist(),
            }
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--activations-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--group",
        default="all",
        help="Activation group tag used by collect_activations.py.",
    )
    p.add_argument(
        "--timestep-evolution-for",
        default="T_plus",
        help="Which actual_group to track across timesteps for D3 analysis.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml_load(args.config)
    analysis_cfg = cfg["analysis"]

    pooled_root = args.activations_root / "activations" / "pooled" / args.group
    if not pooled_root.exists():
        print(f"No pooled activations under {pooled_root}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_root / "analysis" / args.group
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []

    for cell_dir, target, layer, pool in _iter_cells(pooled_root):
        data = load_cell(cell_dir)
        result = analyze_cell(data, analysis_cfg)
        result.update(
            {
                "cell": cell_dir.name,
                "target_ratio": target,
                "layer": layer,
                "pool": pool,
            }
        )
        cell_path = out_dir / f"{cell_dir.name}.json"
        with open(cell_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        summary_rows.append(
            {
                "cell": cell_dir.name,
                "target_ratio": target,
                "layer": layer,
                "pool": pool,
                "n_rows": result["n_rows"],
                "effective_rank": result["effective_rank"],
                "participation_ratio": result["participation_ratio"],
                "k90": result.get("k90"),
                "k95": result.get("k95"),
            }
        )
        print(f"  analyzed {cell_dir.name}", file=sys.stderr)

    # Timestep evolution (optional): iterate over unique (layer, pool) pairs.
    pairs = sorted({(l, p) for _, _, l, p in _iter_cells(pooled_root)})
    evo_out: Dict[str, Any] = {}
    for l, p in pairs:
        evo_out[f"l={l:02d}_p={p}"] = timestep_evolution(
            pooled_root,
            layer=l,
            pool=p,
            group=args.timestep_evolution_for,
            angle_k=int(analysis_cfg.get("principal_angle_max_k", 32)),
        )
    with open(out_dir / "timestep_evolution.json", "w", encoding="utf-8") as fh:
        json.dump(evo_out, fh, indent=2)

    write_jsonl(out_dir / "summary.jsonl", summary_rows)
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
