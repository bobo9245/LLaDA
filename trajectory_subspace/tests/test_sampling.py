"""Sampling-core tests.

Covers:

- Fixed-seed regression parity between the new sampling core and the public
  ``generate.generate`` wrapper.
- Observer path vs no-observer path produce identical token outputs.
- Native ``output_hidden_states`` vs forward-hook fallback produce the same
  hidden-state shapes in the observer-visible layer list.
- Intervention hook actually zeros out the targeted subspace on request.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from trajectory_subspace.observers import CollectObserver
from trajectory_subspace.sampling import (
    NullObserver,
    SamplingConfig,
    reverse_diffusion_sample,
)
from trajectory_subspace.interventions import ProjectionAblation

from trajectory_subspace.tests.fakes import FakeLLaDAModel, NoHiddenFakeLLaDAModel


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_prompt(model: FakeLLaDAModel, length: int = 4) -> torch.Tensor:
    # Any token != mask_id works. Use 5 to be distinct from fill_token=0.
    return torch.full((1, length), 5, dtype=torch.long, device=model.device)


@pytest.fixture
def model():
    m = FakeLLaDAModel(vocab_size=32, hidden_size=16, num_layers=3)
    m.eval()
    return m


# Use an in-vocab mask id so the tiny fake model's embedding lookup works.
_TEST_MASK_ID = 31


@pytest.fixture
def cfg():
    return SamplingConfig(
        steps=8,
        gen_length=8,
        block_length=4,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=_TEST_MASK_ID,
    )


def test_regression_new_vs_old_generate(model, cfg):
    """``generate.generate`` must produce the same sequence as the new core."""
    from generate import generate as wrapper_generate

    prompt = _make_prompt(model)

    _seed_all(1234)
    new_out = reverse_diffusion_sample(model, prompt, cfg)

    _seed_all(1234)
    old_out = wrapper_generate(
        model,
        prompt,
        steps=cfg.steps,
        gen_length=cfg.gen_length,
        block_length=cfg.block_length,
        temperature=cfg.temperature,
        cfg_scale=cfg.cfg_scale,
        remasking=cfg.remasking,
        mask_id=cfg.mask_id,
    )
    assert torch.equal(new_out, old_out)


def test_observer_path_matches_no_observer(model, cfg):
    prompt = _make_prompt(model)

    _seed_all(42)
    out_a = reverse_diffusion_sample(model, prompt, cfg, observer=NullObserver())

    _seed_all(42)
    observer = CollectObserver(
        target_ratios=[0.9, 0.5, 0.1], layers=[0, 1, 2], poolings=["response_mean"]
    )
    cfg_obs = SamplingConfig(
        **{**cfg.__dict__, "capture_layers": (0, 1, 2)}
    )
    out_b = reverse_diffusion_sample(model, prompt, cfg_obs, observer=observer)
    assert torch.equal(out_a, out_b)
    assert observer.records, "CollectObserver should have captured something"


def test_hook_fallback_matches_native():
    native = FakeLLaDAModel(vocab_size=32, hidden_size=16, num_layers=3)
    hook_only = NoHiddenFakeLLaDAModel(vocab_size=32, hidden_size=16, num_layers=3)
    # Copy weights so the two models are numerically identical.
    hook_only.load_state_dict(native.state_dict())

    prompt = torch.full((1, 4), 5, dtype=torch.long)
    cfg = SamplingConfig(
        steps=8,
        gen_length=8,
        block_length=4,
        capture_layers=(0, 1, 2),
        mask_id=_TEST_MASK_ID,
    )

    observers = {}
    for name, mdl in [("native", native), ("hook", hook_only)]:
        obs = CollectObserver(
            target_ratios=[0.5], layers=[0, 1, 2], poolings=["response_mean"]
        )
        _seed_all(99)
        reverse_diffusion_sample(mdl, prompt, cfg, observer=obs)
        observers[name] = obs

    # Tokens match
    for key in observers["native"].records:
        a = observers["native"].records[key].pooled
        b = observers["hook"].records[key].pooled
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-5)


def test_intervention_ablates_subspace(model):
    """Projection ablation should kill the configured subspace in captured acts."""
    prompt = _make_prompt(model)
    basis = torch.zeros(model.hidden_size, 1)
    basis[0, 0] = 1.0  # ablate dim 0
    inter = ProjectionAblation(basis=basis, target_layers=(2,))

    obs = CollectObserver(
        target_ratios=[0.1], layers=[2], poolings=["response_mean"]
    )
    cfg = SamplingConfig(
        steps=8,
        gen_length=8,
        block_length=4,
        capture_layers=(2,),
        mask_id=_TEST_MASK_ID,
    )
    _seed_all(7)
    reverse_diffusion_sample(model, prompt, cfg, observer=obs, intervention=inter)
    key = (0, 2, "response_mean")
    assert key in obs.records
    pooled = obs.records[key].pooled  # (B, H)
    # Dim 0 should be ~0 after ablation (within fp tolerance).
    assert pooled[..., 0].abs().max().item() < 1e-4
