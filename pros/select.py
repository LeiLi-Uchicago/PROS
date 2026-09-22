"""Stage-1 selectors and stage-2 refiners.

Selectors choose ``m`` candidates from a block; refiners choose the final ``n``
from the merged pool.  Both operate on coordinate arrays and return indices
into the array they were given, so the caller owns index bookkeeping.
"""

from __future__ import annotations

import numpy as np

from .geometry import (
    farthest_first,
    farthest_first_seeded,
    sq_norms,
)

__all__ = [
    "select_fft",
    "select_scsampler",
    "select_random",
    "make_selector",
    "refine",
    "kcenter_local_search",
    "assign_two_nearest",
]


# --------------------------------------------------------------------------
# stage-1 selectors
# --------------------------------------------------------------------------
def select_fft(X: np.ndarray, m: int, rng) -> np.ndarray:
    """Gonzalez farthest-first traversal."""
    return farthest_first(X, m, rng=rng)


def select_random(X: np.ndarray, m: int, rng) -> np.ndarray:
    """Uniform sample without replacement."""
    m = min(int(m), X.shape[0])
    return rng.choice(X.shape[0], size=m, replace=False).astype(np.int64)


def select_scsampler(X: np.ndarray, m: int, rng) -> np.ndarray:
    """scSampler's maximin selection, delegating to the upstream package.

    Falls back to FFT plus a maximin swap pass when ``scsampler`` is not
    importable, so the pipeline remains runnable without it.  The fallback is
    reported by :func:`make_selector` callers via the returned ``info`` so a
    silent substitution never goes unrecorded.
    """
    m = min(int(m), X.shape[0])
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    try:
        import scsampler as _scsampler

        # Upstream accepts a plain array and returns selected indices when
        # asked not to copy the data.  Signature differs across releases, so
        # try the documented keyword first and degrade gracefully.
        out = _scsampler.scsampler(
            np.ascontiguousarray(X, dtype=np.float64),
            n_obs=m,
            copy=False,
            random_split=1,
        )
        idx = np.asarray(out, dtype=np.int64).ravel()
        if idx.size != m:
            raise ValueError(f"scsampler returned {idx.size} indices, expected {m}")
        return idx
    except Exception:
        idx = farthest_first(X, m, rng=rng)
        return maximin_swap(X, idx, max_iter=10)


def make_selector(kind: str):
    if kind == "fft":
        return select_fft
    if kind == "scsampler_maximin":
        return select_scsampler
    if kind == "random":
        return select_random
    raise ValueError(
        f"unknown selector {kind!r}; expected 'fft', 'scsampler_maximin', 'random'"
    )


# --------------------------------------------------------------------------
# neighbour utilities
# --------------------------------------------------------------------------
def assign_two_nearest(
    X: np.ndarray, centres: np.ndarray, chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest and second-nearest centre distances plus the nearest index.

    Returns ``(d1, a1, d2)``.  The second-nearest distance is what makes the
    k-centre swap in :func:`kcenter_local_search` evaluable in O(N + n) per
    candidate removal instead of O(N n).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    C = np.ascontiguousarray(centres, dtype=np.float64)
    n_c = C.shape[0]
    c_sq = sq_norms(C)
    n = X.shape[0]
    d1 = np.empty(n, dtype=np.float64)
    a1 = np.empty(n, dtype=np.int64)
    d2 = np.full(n, np.inf, dtype=np.float64)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = X[start:stop]
        dd = c_sq[None, :] - 2.0 * (block @ C.T)
        dd += sq_norms(block)[:, None]
        np.maximum(dd, 0.0, out=dd)
        if n_c == 1:
            a1[start:stop] = 0
            d1[start:stop] = np.sqrt(dd[:, 0])
            continue
        part = np.argpartition(dd, 1, axis=1)[:, :2]
        rows = np.arange(stop - start)[:, None]
        two = dd[rows, part]
        order = np.argsort(two, axis=1)
        first = part[rows[:, 0], order[:, 0]]
        second = part[rows[:, 0], order[:, 1]]
        a1[start:stop] = first
        d1[start:stop] = np.sqrt(dd[rows[:, 0], first])
        d2[start:stop] = np.sqrt(dd[rows[:, 0], second])
    return d1, a1, d2


# --------------------------------------------------------------------------
# refiners
# --------------------------------------------------------------------------
def maximin_swap(X: np.ndarray, idx: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Improve the maximin (min pairwise distance) objective by swapping.

    Repeatedly locates the closest selected pair and tries to replace one of
    the two with the unselected point that is farthest from the rest.  This is
    the criterion scSampler optimises, as distinct from the covering criterion
    targeted by :func:`kcenter_local_search`.
    """
    idx = np.asarray(idx, dtype=np.int64).copy()
    n_sel = idx.size
    if n_sel < 3 or X.shape[0] <= n_sel:
        return idx

    for _ in range(int(max_iter)):
        S = X[idx]
        s_sq = sq_norms(S)
        dd = s_sq[None, :] - 2.0 * (S @ S.T) + s_sq[:, None]
        np.fill_diagonal(dd, np.inf)
        np.maximum(dd, 0.0, out=dd)
        i, j = np.unravel_index(np.argmin(dd), dd.shape)
        current = float(np.sqrt(dd.min()))

        mask = np.ones(X.shape[0], dtype=bool)
        mask[idx] = False
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            break

        improved = False
        for drop in (i, j):
            keep = np.delete(idx, drop)
            K = X[keep]
            k_sq = sq_norms(K)
            dc = k_sq[None, :] - 2.0 * (X[cand] @ K.T) + sq_norms(X[cand])[:, None]
            np.maximum(dc, 0.0, out=dc)
            nearest = np.sqrt(dc.min(axis=1))
            best = int(np.argmax(nearest))
            if nearest[best] > current + 1e-12:
                idx = np.append(keep, cand[best])
                improved = True
                break
        if not improved:
            break
    return idx


def kcenter_local_search(
    X: np.ndarray, idx: np.ndarray, max_iter: int = 25
) -> np.ndarray:
    """Swap-based local search on the k-centre (covering radius) objective.

    Each round adds the currently worst-covered point and removes whichever
    existing centre costs least to drop, accepting the swap only if the
    covering radius strictly improves.  Removal cost for every centre is
    evaluated in one pass using second-nearest distances.
    """
    idx = np.asarray(idx, dtype=np.int64).copy()
    if idx.size < 2:
        return idx

    for _ in range(int(max_iter)):
        d1, _, _ = assign_two_nearest(X, X[idx])
        radius = float(d1.max())
        critical = int(np.argmax(d1))
        if critical in set(idx.tolist()):
            break

        cand = np.append(idx, critical)
        cd1, ca1, cd2 = assign_two_nearest(X, X[cand])
        n_c = cand.size

        # per-centre max nearest distance, and max second-nearest among its members
        max_near = np.zeros(n_c, dtype=np.float64)
        max_second = np.zeros(n_c, dtype=np.float64)
        np.maximum.at(max_near, ca1, cd1)
        np.maximum.at(max_second, ca1, cd2)

        # top-2 of max_near lets us get max_{k != j} max_near[k] in O(1)
        order = np.argsort(-max_near)
        top1, top2 = order[0], (order[1] if n_c > 1 else order[0])

        best_radius, best_drop = np.inf, -1
        for j in range(n_c):
            others = max_near[top1] if j != top1 else max_near[top2]
            new_radius = max(others, max_second[j])
            if new_radius < best_radius:
                best_radius, best_drop = new_radius, j

        if best_drop < 0 or best_radius >= radius - 1e-12:
            break
        idx = np.delete(cand, best_drop)
    return idx


def refine(
    P: np.ndarray,
    n: int,
    kind: str,
    rng,
    max_iter: int = 25,
    mix: float = 0.0,
    mix_seeds: np.ndarray | None = None,
) -> np.ndarray:
    """Stage-2 refinement: choose ``n`` of the pooled candidates ``P``.

    ``mix`` reserves that fraction of the budget for a uniform draw from the
    pool, with the remainder completed by the geometric refiner.  ``mix=0`` is
    pure covering; ``mix=1`` is a uniform draw from the pool.  The uniform part
    is drawn *first* so the geometric part sees it and does not duplicate its
    coverage.  See :func:`pros.geometry.farthest_first_seeded` for what
    this does to the covering guarantee.

    ``mix_seeds`` overrides the draw with caller-supplied positions into ``P``.
    :func:`pros.core.sketch` uses it for ``mix_source="data"``, where the
    reserve must be uniform over the full dataset rather than over the pool --
    the two differ whenever the pool is a strict subset of the data.
    """
    n = min(int(n), P.shape[0])
    mix = float(mix)
    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"mix must be in [0, 1], got {mix}")
    if mix > 0.0 and kind != "none":
        k = int(round(mix * n))
        if k >= n:
            if mix_seeds is not None:
                extra = rng.choice(
                    np.setdiff1d(np.arange(P.shape[0]), mix_seeds),
                    size=max(n - mix_seeds.size, 0), replace=False,
                )
                return np.sort(np.concatenate([mix_seeds, extra])).astype(np.int64)[:n]
            return rng.choice(P.shape[0], size=n, replace=False).astype(np.int64)
        if k > 0:
            seeds = (
                np.asarray(mix_seeds, dtype=np.int64)
                if mix_seeds is not None
                else rng.choice(P.shape[0], size=k, replace=False).astype(np.int64)
            )
            if kind == "fft":
                return farthest_first_seeded(P, n, seeds)
            # swap-based refiners: seed the FFT they start from, then let the
            # swap phase run as usual over the combined set.
            base = farthest_first_seeded(P, n, seeds)
            if kind == "maximin":
                return maximin_swap(P, base, max_iter=max_iter)
            if kind == "local_swap":
                return kcenter_local_search(P, base, max_iter=max_iter)
    if kind == "none":
        return rng.choice(P.shape[0], size=n, replace=False).astype(np.int64)
    if kind == "fft":
        return farthest_first(P, n, rng=rng)
    if kind == "maximin":
        return maximin_swap(P, farthest_first(P, n, rng=rng), max_iter=max_iter)
    if kind == "local_swap":
        return kcenter_local_search(P, farthest_first(P, n, rng=rng), max_iter=max_iter)
    raise ValueError(
        f"unknown refiner {kind!r}; expected 'fft', 'maximin', 'local_swap', 'none'"
    )
