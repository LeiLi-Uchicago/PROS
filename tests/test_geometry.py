from __future__ import annotations

import numpy as np

from pros.geometry import (
    FarthestFirst,
    covering_radius,
    farthest_first,
    min_pairwise_distance,
)


def test_covering_radius_is_zero_for_full_set() -> None:
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    assert covering_radius(X, X) == 0.0


def test_farthest_first_selects_requested_count() -> None:
    X = np.random.default_rng(0).random((40, 3))
    idx = farthest_first(X, 7, seed_index=0)

    assert idx.shape == (7,)
    assert np.unique(idx).size == 7


def test_farthest_first_has_zero_radius_after_selecting_every_row() -> None:
    X = np.random.default_rng(9).random((17, 4))
    state = FarthestFirst(X, rng=np.random.default_rng(3))
    state.run_to(X.shape[0])

    assert state.radius == 0.0


def test_min_pairwise_distance() -> None:
    X = np.array([[0.0], [1.0], [5.0]])
    assert min_pairwise_distance(X) == 1.0
