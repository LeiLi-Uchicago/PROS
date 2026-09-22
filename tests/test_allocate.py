from __future__ import annotations

import numpy as np

from pros.allocate import make_allocation, water_filling


def test_allocation_sums_to_budget() -> None:
    X = np.random.default_rng(0).random((90, 3))
    labels = np.repeat([0, 1, 2], 30)

    for kind in ["proportional", "power", "volume", "uniform"]:
        m = make_allocation(X, labels, 3, 24, kind, d_intrinsic=2)
        assert int(m.sum()) == 24
        assert np.all(m <= 30)


def test_water_filling_returns_pool_and_counts() -> None:
    X = np.random.default_rng(1).random((60, 3))
    labels = np.repeat([0, 1, 2], 20)

    pool, counts, info = water_filling(X, labels, 3, 18, np.random.default_rng(1))

    assert pool.shape == (18,)
    assert int(counts.sum()) == 18
    assert "rho_blockwise_max" in info


def test_water_filling_conserves_budget_with_empty_blocks() -> None:
    X = np.random.default_rng(5).random((5, 2))
    labels = np.array([0, 0, 0, 3, 3], dtype=np.int32)

    pool, counts, info = water_filling(X, labels, 5, 4, np.random.default_rng(2))

    assert pool.size == 4
    assert np.unique(pool).size == 4
    assert np.all((0 <= pool) & (pool < X.shape[0]))
    assert int(counts.sum()) == 4
    assert counts[1] == counts[2] == counts[4] == 0
    assert info["block_radii"].shape == (5,)


def test_allocators_respect_caps_with_empty_blocks() -> None:
    X = np.random.default_rng(6).random((5, 2))
    labels = np.array([0, 0, 0, 3, 3], dtype=np.int32)

    for kind in ["proportional", "power", "volume", "uniform"]:
        counts = make_allocation(X, labels, 5, 5, kind, d_intrinsic=2)
        assert int(counts.sum()) == 5
        assert np.all(counts >= 0)
        assert np.array_equal(counts[[1, 2, 4]], np.zeros(3, dtype=np.int64))
