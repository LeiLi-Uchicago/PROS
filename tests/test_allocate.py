from __future__ import annotations

import numpy as np

from pros import make_allocation, water_filling


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

