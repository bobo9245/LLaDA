"""Pure analysis primitives on synthetic fixtures.

We construct matrices with a known low-rank structure and verify that:

- ``k90`` / ``k95`` recover the injected rank.
- ``participation_ratio`` sits close to the true rank.
- ``principal_angles`` are small between subspaces spanned by aligned data.
- ``held_out_reconstruction`` is close to 1.0 for the true rank when
  train/test share the same low-rank structure.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trajectory_subspace.scripts.analyze_subspaces import (
    effective_rank,
    held_out_reconstruction,
    k_for_cumulative,
    participation_ratio,
    principal_angles,
    svd_spectrum,
    top_k_basis,
)


def _low_rank(n: int, h: int, rank: int, seed: int, noise: float = 1e-3):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, rank))
    V = rng.normal(size=(rank, h))
    X = U @ V
    X = X + rng.normal(size=X.shape) * noise * np.linalg.norm(X) / (np.sqrt(n * h))
    return X.astype(np.float64)


def test_k90_recovers_rank():
    X = _low_rank(200, 32, rank=4, seed=0, noise=1e-4)
    s = svd_spectrum(X)
    assert k_for_cumulative(s, 0.9) == 4
    assert k_for_cumulative(s, 0.95) == 4


def test_participation_ratio_close_to_rank():
    X = _low_rank(200, 32, rank=4, seed=1, noise=1e-5)
    s = svd_spectrum(X)
    pr = participation_ratio(s)
    assert 3.5 < pr < 4.5


def test_effective_rank_close_to_log_rank():
    X = _low_rank(200, 32, rank=4, seed=2, noise=1e-5)
    s = svd_spectrum(X)
    er = effective_rank(s)
    # Entropy of uniform over 4 modes is log(4), exp(...) ~= 4.
    assert 3.0 < er < 5.0


def test_held_out_reconstruction_close_to_one():
    X = _low_rank(200, 32, rank=3, seed=3, noise=1e-5)
    train = X[:150]
    test = X[150:]
    caps = held_out_reconstruction(train, test, ks=[1, 2, 3, 4])
    assert caps[3] > 0.99
    assert caps[1] < caps[3]


def test_principal_angles_aligned_vs_random():
    Xa = _low_rank(200, 32, rank=3, seed=4, noise=1e-5)
    Xb = Xa + np.random.default_rng(5).normal(size=Xa.shape) * 1e-6
    Ba = top_k_basis(Xa, 3)
    Bb = top_k_basis(Xb, 3)
    aligned = principal_angles(Ba, Bb)
    # Aligned subspaces should share ~same basis.
    assert aligned.max() < 1e-2

    Xc = _low_rank(200, 32, rank=3, seed=7, noise=1e-5)
    Bc = top_k_basis(Xc, 3)
    rnd = principal_angles(Ba, Bc)
    # Random subspaces in 32d sit at non-trivial angles.
    assert rnd.max() > 0.5
