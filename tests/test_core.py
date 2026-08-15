from __future__ import annotations

import numpy as np
import pytest

from pros import sketch


def test_sketch_returns_unique_indices() -> None:
    X = np.random.default_rng(0).random((300, 6))
    res = sketch(X, n=30, n_blocks=8, seed=0)

    assert res.indices.shape == (30,)
    assert res.indices.dtype == np.int64
    assert np.unique(res.indices).size == 30
    assert res.pool_indices.size >= 30
    assert set(res.timings) >= {"partition", "stage1", "stage2", "total"}


def test_sketch_certificate_fields() -> None:
    X = np.random.default_rng(1).random((120, 4))
    res = sketch(X, n=12, n_blocks=4, r=3, certify=True, seed=1)

    assert res.radius >= 0
    assert res.rho >= 0
    assert res.opt_lower >= 0
    assert res.ratio_upper >= 1


def test_sketch_validates_inputs() -> None:
    X = np.random.default_rng(2).random((20, 3))

    with pytest.raises(ValueError, match="n must be"):
        sketch(X, n=0)
    with pytest.raises(ValueError, match="oversampling ratio"):
        sketch(X, n=5, r=0.5)
    with pytest.raises(ValueError, match="mix_source"):
        sketch(X, n=5, mix_source="bad")

