"""End-to-end smoke test of the manifest -> activation -> analysis pipeline.

Uses the tiny fake LLaDA model from ``fakes`` to avoid any GPU / weight
dependency. Covers:

- ShardWriter resume (index rows persist across writer instances).
- CollectObserver captures pooled activations keyed by target ratio.
- ``load_cell`` reads back what ``ShardWriter`` wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from trajectory_subspace.io_utils import ShardWriter, load_cell, read_jsonl
from trajectory_subspace.observers import CollectObserver
from trajectory_subspace.sampling import SamplingConfig, reverse_diffusion_sample
from trajectory_subspace.tests.fakes import FakeLLaDAModel


pytest.importorskip("safetensors")


def test_shard_writer_roundtrip(tmp_path: Path):
    cell_dir = tmp_path / "cell"
    writer = ShardWriter(cell_dir, shard_max_rows=2)
    for i in range(5):
        writer.append(torch.full((4,), float(i)), {"prompt_id": f"p{i}", "seed": 0})
    writer.close()

    rows = list(read_jsonl(cell_dir / "index.jsonl"))
    assert len(rows) == 5
    assert {r["prompt_id"] for r in rows} == {f"p{i}" for i in range(5)}

    loaded = load_cell(cell_dir)
    assert loaded["values"].shape == (5, 4)
    for i, r in enumerate(loaded["meta"]):
        assert torch.allclose(loaded["values"][i], torch.full((4,), float(i)))


def test_shard_writer_resume(tmp_path: Path):
    cell_dir = tmp_path / "cell"
    writer = ShardWriter(cell_dir, shard_max_rows=2)
    for i in range(3):
        writer.append(torch.full((4,), float(i)), {"prompt_id": f"p{i}", "seed": 0})
    writer.close()

    # Re-open: index rows should persist and new shards continue the count.
    writer2 = ShardWriter(cell_dir, shard_max_rows=2)
    assert writer2._row_cursor == 3
    assert writer2.already_has(lambda r: r.get("prompt_id") == "p1")
    writer2.append(torch.full((4,), 9.0), {"prompt_id": "p9", "seed": 0})
    writer2.close()

    loaded = load_cell(cell_dir)
    assert loaded["values"].shape == (4, 4)


def test_pipeline_smoke(tmp_path: Path):
    model = FakeLLaDAModel(vocab_size=32, hidden_size=16, num_layers=3)
    prompt = torch.full((1, 4), 5, dtype=torch.long)
    cfg = SamplingConfig(
        steps=8,
        gen_length=8,
        block_length=4,
        capture_layers=(1, 2),
        mask_id=31,  # In-vocab mask id for the fake model.
    )
    obs = CollectObserver(
        target_ratios=[0.8, 0.4, 0.1],
        layers=[1, 2],
        poolings=["response_mean", "last_unmasked"],
    )
    reverse_diffusion_sample(model, prompt, cfg, observer=obs)

    cell_root = tmp_path / "cells"
    total = 0
    for (ti, layer, pool), rec in obs.records.items():
        cell_dir = cell_root / f"t={ti}_l={layer:02d}_p={pool}"
        writer = ShardWriter(cell_dir, shard_max_rows=4)
        for b in range(rec.pooled.shape[0]):
            writer.append(rec.pooled[b], {"prompt_id": "p0", "seed": 0,
                                           "target_ratio_idx": ti,
                                           "layer": layer, "pool": pool,
                                           "actual_group": "T_plus"})
            total += 1
        writer.close()
    assert total > 0

    # Each cell should be readable.
    for cell_dir in cell_root.iterdir():
        loaded = load_cell(cell_dir)
        assert loaded["values"].shape[0] >= 1
        assert loaded["values"].shape[1] == 16
