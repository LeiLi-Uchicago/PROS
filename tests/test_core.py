from __future__ import annotations

import numpy as np
import pytest

from pros import sketch


def assert_valid_result(result, n: int, n_rows: int) -> None:
    assert result.indices.shape == (n,)
    assert result.indices.dtype == np.int64
    assert np.unique(result.indices).size == n
    assert np.all((0 <= result.indices) & (result.indices < n_rows))
    assert result.pool_indices.size >= n
    assert np.unique(result.pool_indices).size == result.pool_indices.size
    assert np.all((0 <= result.pool_indices) & (result.pool_indices < n_rows))
    assert set(result.timings) >= {"partition", "stage1", "stage2", "total"}


def test_sketch_is_reproducible_and_returns_valid_indices() -> None:
    X = np.random.default_rng(0).random((300, 6))
    first = sketch(X, n=30, n_blocks=8, seed=11)
    second = sketch(X, n=30, n_blocks=8, seed=11)

    assert_valid_result(first, 30, X.shape[0])
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.pool_indices, second.pool_indices)


@pytest.mark.parametrize("partitioner", ["random", "kmeans", "pc_tree", "none"])
def test_small_inputs_and_excess_block_requests_are_feasible(partitioner: str) -> None:
    X = np.column_stack([np.arange(5, dtype=float), np.ones(5)])
    result = sketch(X, n=3, partitioner=partitioner, n_blocks=20, r=2, seed=4)

    assert_valid_result(result, 3, X.shape[0])
    assert result.n_blocks <= X.shape[0]


def test_duplicate_points_and_constant_dimensions_are_supported() -> None:
    X = np.array(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    )
    result = sketch(
        X, n=3, partitioner="pc_tree", n_blocks=8, r=10, seed=3, certify=True
    )

    assert_valid_result(result, 3, X.shape[0])
    assert result.radius == 0.0
    assert result.opt_lower == 0.0
    assert np.isnan(result.ratio_upper)


def test_n_equal_to_population_returns_every_index() -> None:
    X = np.random.default_rng(2).random((12, 3))
    result = sketch(X, n=X.shape[0], n_blocks=30, r=10, seed=1, certify=True)

    assert np.array_equal(result.indices, np.arange(X.shape[0]))
    assert np.array_equal(result.pool_indices, np.arange(X.shape[0]))
    assert result.budget == X.shape[0]
    assert result.opt_lower == 0.0
    assert np.isnan(result.ratio_upper)


def test_oversampling_budget_is_capped_by_population() -> None:
    X = np.random.default_rng(3).random((20, 3))
    result = sketch(X, n=7, partitioner="random", n_blocks=4, r=2.25, seed=1)
    saturated = sketch(X, n=7, partitioner="random", n_blocks=4, r=10, seed=1)

    assert result.budget == 16
    assert result.pool_indices.size == 16
    assert saturated.budget == X.shape[0]
    assert saturated.pool_indices.size == X.shape[0]


def test_sketch_validates_inputs() -> None:
    X = np.random.default_rng(2).random((20, 3))

    with pytest.raises(ValueError, match="n must be"):
        sketch(X, n=0)
    with pytest.raises(ValueError, match="oversampling ratio"):
        sketch(X, n=5, r=0.5)
    with pytest.raises(ValueError, match="mix_source"):
        sketch(X, n=5, mix_source="bad")
    with pytest.raises(ValueError, match="n_blocks"):
        sketch(X, n=5, n_blocks=0)
    with pytest.raises(ValueError, match="2-D"):
        sketch(np.arange(10), n=5)
