from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from pros import certificate, opt_bounds
from pros.geometry import covering_radius


def exact_optimum_radius(X: np.ndarray, n: int) -> float:
    return min(
        covering_radius(X, X[list(indices)])
        for indices in combinations(range(X.shape[0]), n)
    )


def test_opt_bounds_and_certificate_are_valid_against_exact_small_problem() -> None:
    X = np.array([[0.0], [1.0], [4.0], [10.0]])
    n = 2
    optimum = exact_optimum_radius(X, n)
    bounds = opt_bounds(X, n, seed=2)
    cert = certificate(
        X, np.array([0, 3]), pool_indices=np.arange(X.shape[0]), opt_cache=bounds
    )

    assert bounds["opt_lower"] <= optimum + 1e-12
    assert bounds["opt_upper"] >= optimum - 1e-12
    assert cert["ratio_upper"] >= cert["radius"] / optimum - 1e-12
    assert cert["rho"] == 0.0
    assert cert["theory_bound"] >= cert["radius"]


def test_certificate_accepts_reusable_matching_cache() -> None:
    X = np.random.default_rng(7).random((15, 3))
    cache = opt_bounds(X, 4, seed=1)
    cert = certificate(X, np.array([0, 2, 7, 11]), opt_cache=cache)

    assert cert["n"] == 4
    assert cert["opt_lower"] == cache["opt_lower"]


@pytest.mark.parametrize(
    "indices, message",
    [
        (np.array([], dtype=np.int64), "non-empty"),
        (np.array([0, 0]), "duplicates"),
        (np.array([-1, 1]), "valid indices"),
    ],
)
def test_certificate_rejects_invalid_sketch_indices(
    indices: np.ndarray, message: str
) -> None:
    X = np.random.default_rng(8).random((8, 2))
    with pytest.raises(ValueError, match=message):
        certificate(X, indices)


def test_certificate_rejects_mismatched_cache() -> None:
    X = np.random.default_rng(9).random((8, 2))
    with pytest.raises(ValueError, match="opt_cache"):
        certificate(X, np.array([0, 2, 4]), opt_cache=opt_bounds(X, 2))
