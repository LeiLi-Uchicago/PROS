"""Near-optimality certificates for a sketch.

What is certifiable, and at what cost
-------------------------------------
For the k-centre objective ``R(X, S) = max_i min_{s in S} d(x_i, s)`` we can
compute, *at run time and without knowing the optimum*:

``opt_lower``
    A valid lower bound on ``OPT_n(X)``.  A farthest-first traversal of
    ``n + 1`` points yields ``n + 1`` cells that are pairwise at least
    ``delta`` apart, where ``delta`` is the final hop distance.  Any ``n``
    centres must cover two of them with one centre, so
    ``OPT_n >= delta / 2``.

``ratio_upper``
    ``R(X, S) / opt_lower`` -- a *valid upper bound* on the true approximation
    ratio ``R(X, S) / OPT_n``.  This is the headline certificate: a run that
    reports ``ratio_upper = 2.4`` has proven it is within 2.4x of optimal on
    that dataset, whatever the optimum happens to be.

Honest accounting of cost
-------------------------
Obtaining ``delta`` requires a farthest-first traversal over the full data.
Certification is therefore opt-in; its separate costs are reported in
``timings``. :func:`choose_r` provides a subsampling-based estimate for
choosing an oversampling ratio.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from ._validation import indices, integer, matrix
from .geometry import FarthestFirst, covering_radius

__all__ = ["certificate", "opt_bounds", "choose_r", "estimate_opt_scale"]


def _fingerprint(X):
    digest = hashlib.sha256(str(X.shape).encode())
    digest.update(memoryview(X).cast("B"))
    return digest.hexdigest()


def opt_bounds(X: np.ndarray, n: int, seed: int = 0) -> dict:
    """Lower and upper bounds on the optimal ``n``-centre radius of ``X``.

    Returns ``opt_lower = delta / 2`` and ``opt_upper = R_fft(n)``, both derived
    from a single farthest-first traversal to ``n + 1`` points.  For Gonzalez's
    traversal these two numbers differ by exactly a factor of two, which is the
    classical 2-approximation.
    """
    X = matrix(X)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array")
    n = integer(n, "n", minimum=1, maximum=len(X))
    if not 0 < n <= X.shape[0]:
        raise ValueError(f"n must be in (0, {X.shape[0]}], got {n}")
    rng = np.random.default_rng(seed)
    state = FarthestFirst(X, rng=rng)
    state.run_to(min(n + 1, X.shape[0]))

    # radii[k-1] is the covering radius achieved by the first k points.
    radius_n = float(state.radii[n - 1])
    # The (n+1)-th hop distance equals the covering radius after n points, and
    # is the minimum pairwise distance among the n+1 selected points.
    delta = radius_n
    return {
        "data_fingerprint": _fingerprint(X),
        "opt_lower": delta / 2.0,
        "opt_upper": radius_n,
        "fft_radius_n": radius_n,
        "delta": delta,
        "fft_indices": np.asarray(state.selected[:n], dtype=np.int64),
    }


def certificate(
    X: np.ndarray,
    sketch_indices: np.ndarray,
    pool_indices: np.ndarray | None = None,
    seed: int = 0,
    opt_cache: dict | None = None,
    *,
    assume_fft_refinement: bool = False,
) -> dict:
    """Full certificate for one sketch.

    Parameters
    ----------
    opt_cache
        Result of a previous :func:`opt_bounds` call at the same ``n``.  The
        bounds depend only on ``(X, n)``, not on the method under test, so one
        computation is shared across every method at that sketch size.

    Returns
    -------
    dict
        ``radius``, ``rho``, ``opt_lower``, ``opt_upper``, ``ratio_upper``
        (certified), ``theory_bound`` (``2*opt_upper + 3*rho``),
        ``bound_slack`` (``radius / theory_bound``), and ``timings``.
        The theory fields are NaN unless assume_fft_refinement=True. This
        flag asserts unseeded FFT on the supplied pool; it is not verified
        from arbitrary user-provided indices. ratio_upper is independent of it.
        Floating-point distances are numerical estimates, not interval-certified bounds.
    """
    X = matrix(X)
    sketch_indices = indices(sketch_indices, len(X), "sketch_indices")
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array")
    if sketch_indices.ndim != 1 or sketch_indices.size == 0:
        raise ValueError("sketch_indices must be a non-empty one-dimensional array")
    if np.any(sketch_indices < 0) or np.any(sketch_indices >= X.shape[0]):
        raise ValueError("sketch_indices must be valid indices into X")
    if np.unique(sketch_indices).size != sketch_indices.size:
        raise ValueError("sketch_indices must not contain duplicates")
    if pool_indices is not None:
        pool_indices = indices(pool_indices, len(X), "pool_indices")
        if not np.isin(sketch_indices, pool_indices).all():
            raise ValueError("sketch must be a subset of pool_indices")
        if pool_indices.ndim != 1 or pool_indices.size == 0:
            raise ValueError("pool_indices must be a non-empty one-dimensional array")
        if np.any(pool_indices < 0) or np.any(pool_indices >= X.shape[0]):
            raise ValueError("pool_indices must be valid indices into X")
        if np.unique(pool_indices).size != pool_indices.size:
            raise ValueError("pool_indices must not contain duplicates")
    n = sketch_indices.size
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    bounds = opt_cache if opt_cache is not None else opt_bounds(X, n, seed=seed)
    if int(np.asarray(bounds.get("fft_indices", [])).size) != n:
        raise ValueError(
            "opt_cache must come from opt_bounds(X, n) for this sketch size"
        )
    if bounds.get("data_fingerprint") != _fingerprint(X):
        raise ValueError(
            "opt_cache belongs to different data or an older unsupported format"
        )
    timings["opt_bounds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    radius = covering_radius(X, X[sketch_indices])
    rho = (
        covering_radius(X, X[pool_indices])
        if pool_indices is not None
        else float("nan")
    )
    timings["radii"] = time.perf_counter() - t0

    opt_lower = float(bounds["opt_lower"])
    opt_upper = float(bounds["opt_upper"])
    ratio_upper = (
        radius / opt_lower
        if opt_lower > 0
        else (float("inf") if radius > 0 else float("nan"))
    )
    theory_bound = (
        2.0 * opt_upper + 3.0 * rho
        if assume_fft_refinement and np.isfinite(rho)
        else float("nan")
    )

    return {
        "n": n,
        "radius": radius,
        "rho": rho,
        "opt_lower": opt_lower,
        "opt_upper": opt_upper,
        "ratio_upper": ratio_upper,
        "theory_bound": theory_bound,
        "bound_slack": (
            radius / theory_bound if theory_bound and theory_bound > 0 else float("nan")
        ),
        "rho_over_opt": rho / opt_lower if opt_lower > 0 else float("nan"),
        "timings": timings,
    }


def estimate_opt_scale(
    X: np.ndarray,
    n: int,
    subsample: int = 30_000,
    seed: int = 0,
) -> float:
    """Cheap *estimate* (not a bound) of the target covering radius.

    Runs a farthest-first traversal on a uniform subsample.  Because a
    subsample's covering radius understates the full data's, this is biased low
    and is only used to set a stopping threshold in :func:`choose_r`.
    """
    X = matrix(X)
    n = integer(n, "n", minimum=1, maximum=len(X))
    subsample = integer(subsample, "subsample", minimum=1)
    rng = np.random.default_rng(seed)
    n_cells = X.shape[0]
    if n_cells > subsample:
        idx = rng.choice(n_cells, size=subsample, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    n_eff = int(min(n, Xs.shape[0]))
    state = FarthestFirst(Xs, rng=rng)
    state.run_to(n_eff)
    return float(state.radii[n_eff - 1])


def choose_r(
    X: np.ndarray,
    n: int,
    tol: float = 0.5,
    r_grid: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 12.0, 20.0),
    seed: int = 0,
    **sketch_kwargs,
) -> dict:
    """Pick the smallest oversampling ratio whose pool is fine enough.

    Grows the stage-1 pool until the measured pool covering radius ``rho``
    falls below ``tol`` times the estimated target radius, turning ``r`` from a
    hyperparameter the user must tune into an accuracy tolerance they request.

    Returns the chosen ``r``, the ``rho`` trajectory, and the accepted sketch.
    """
    from .core import sketch as _sketch

    X = matrix(X)
    n = integer(n, "n", minimum=1, maximum=len(X))
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    r_grid = tuple(float(r) for r in r_grid)
    if not r_grid or any(not np.isfinite(r) or r < 1 for r in r_grid):
        raise ValueError("r_grid must contain finite ratios >= 1")
    if any(a >= b for a, b in zip(r_grid, r_grid[1:], strict=False)):
        raise ValueError("r_grid must be strictly increasing")
    if {"r", "certify"} & sketch_kwargs.keys():
        raise ValueError("choose_r controls r and certify")
    target = estimate_opt_scale(X, n, seed=seed)
    trajectory = []
    accepted = False
    for r in r_grid:
        res = _sketch(X, n, r=r, seed=seed, certify=False, **sketch_kwargs)
        rho = covering_radius(X, X[res.pool_indices])
        trajectory.append(
            {"r": r, "rho": rho, "ratio": rho / target if target > 0 else np.nan}
        )
        if rho <= tol * target:
            accepted = True
            break
    # Only certify the final returned sketch, not every trial.
    cert = certificate(
        X,
        res.indices,
        pool_indices=res.pool_indices,
        seed=seed,
        assume_fft_refinement=(
            res.config["refiner"] == "fft" and res.config["mix"] == 0
        ),
    )
    cert_times = cert.pop("timings")
    res.update(cert)
    res.timings.update({"certify_" + k: v for k, v in cert_times.items()})
    res.config["certify"] = True
    return {
        "r": r,
        "sketch": res,
        "target_radius": target,
        "trajectory": trajectory,
        "tol": tol,
        "tolerance_met": accepted,
    }
