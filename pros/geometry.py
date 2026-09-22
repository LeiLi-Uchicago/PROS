"""Low-level geometric primitives for sketching.

The workhorse is :class:`FarthestFirst`, an *incremental* Gonzalez
farthest-first traversal (FFT).  Exposing the traversal as a resumable object
(rather than a one-shot function) is what lets the water-filling allocator in
:mod:`pros.allocate` reuse exactly the same code path as a plain
single-block FFT: it simply keeps one ``FarthestFirst`` per block in a priority
queue and advances whichever block currently has the largest covering radius.

Distance computations use the BLAS identity

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>

so each traversal step is a single matrix-vector product (``X @ X[j]``) rather
than a broadcast subtraction.  The broadcast form ``((X - X[j])**2).sum(1)``
allocates an ``N x d`` temporary on every step and is roughly an order of
magnitude slower for the array shapes used here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sq_norms",
    "FarthestFirst",
    "farthest_first",
    "farthest_first_seeded",
    "covering_radius",
    "min_pairwise_distance",
    "assign_nearest",
]


def sq_norms(X: np.ndarray) -> np.ndarray:
    """Row-wise squared Euclidean norms of ``X``."""
    return np.einsum("ij,ij->i", X, X)


class FarthestFirst:
    """Incremental Gonzalez farthest-first traversal over a fixed point set.

    Parameters
    ----------
    X
        ``(n_points, n_features)`` array.  Cast to C-contiguous float64.
    seed_index
        Index of the first selected point.  If ``None``, drawn from ``rng``
        (or 0 when ``rng`` is also ``None``).
    rng
        ``numpy.random.Generator`` used only to pick the seed.

    Notes
    -----
    ``radius`` is the *current* covering radius of the selected set over ``X``,
    i.e. ``max_i min_{s in S} d(x_i, s)``.  Because FFT always selects the
    point realising that maximum, the hop distance of the (k+1)-th point equals
    the radius after k points, and the sequence of radii is non-increasing.
    That monotonicity is what makes :func:`min_pairwise_distance` on an FFT
    prefix equal to the radius, and underpins the ``OPT >= delta / 2`` lower
    bound used by :mod:`pros.certify`.
    """

    def __init__(self, X: np.ndarray, seed_index: int | None = None, rng=None):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        self.n_points = self.X.shape[0]
        if self.n_points == 0:
            raise ValueError("X must contain at least one point")
        self._sq = sq_norms(self.X)
        self.mindist2 = np.full(self.n_points, np.inf, dtype=np.float64)
        self.selected: list[int] = []
        # Radius *before* each selection, i.e. hop distances. radii[k] is the
        # covering radius achieved by the first k+1 selected points.
        self.radii: list[float] = []

        if seed_index is None:
            seed_index = int(rng.integers(self.n_points)) if rng is not None else 0
        self._add(int(seed_index))

    # -- internals ---------------------------------------------------------
    def _add(self, j: int) -> None:
        d2 = self._sq + self._sq[j] - 2.0 * (self.X @ self.X[j])
        np.maximum(d2, 0.0, out=d2)  # guard against round-off negatives
        # A selected row is exactly covered by itself. The BLAS identity can
        # otherwise leave a tiny positive residual when every row is selected.
        d2[j] = 0.0
        np.minimum(self.mindist2, d2, out=self.mindist2)
        self.selected.append(int(j))
        self.radii.append(float(np.sqrt(self.mindist2.max())))

    # -- public API --------------------------------------------------------
    @property
    def n_selected(self) -> int:
        return len(self.selected)

    @property
    def radius(self) -> float:
        """Covering radius of the currently selected set over ``X``."""
        return self.radii[-1]

    @property
    def exhausted(self) -> bool:
        return len(self.selected) >= self.n_points

    def hop(self) -> tuple[int, float]:
        """Select the point farthest from the current set.

        Returns ``(index, hop_distance)`` where ``hop_distance`` is the
        distance from the newly selected point to the previously selected set
        (equivalently, the covering radius *before* this hop).
        """
        if self.exhausted:
            raise RuntimeError("all points have been selected")
        j = int(np.argmax(self.mindist2))
        hop_distance = float(np.sqrt(self.mindist2[j]))
        self._add(j)
        return j, hop_distance

    def add(self, j: int) -> None:
        """Force-select point ``j``, updating the covering state.

        Used to pre-seed a traversal (see :func:`farthest_first_seeded`).  No-op
        if ``j`` is already selected.
        """
        if j in self.selected:
            return
        self._add(int(j))

    def run_to(self, n: int) -> list[int]:
        """Advance the traversal until ``n`` points are selected."""
        target = min(int(n), self.n_points)
        while self.n_selected < target:
            self.hop()
        return self.selected


def farthest_first(
    X: np.ndarray,
    n: int,
    seed_index: int | None = None,
    rng=None,
    return_state: bool = False,
):
    """One-shot farthest-first traversal returning ``n`` indices into ``X``.

    Gonzalez's greedy k-centre algorithm: repeatedly take the point furthest
    from everything already chosen.  It is a 2-approximation to the optimal
    covering radius, and costs ``O(N * n)`` distance evaluations -- which is
    exactly the cost PROS avoids by running this over a pool of size ``r*n``
    instead of over all ``N`` points.

    Parameters
    ----------
    X : numpy.ndarray
        ``(N, d)`` coordinates.
    n : int
        Number of points to select.
    seed_index : int, optional
        Index of the first point.  If None, drawn from ``rng``.
    rng : numpy.random.Generator, optional
        Source of randomness for the first point only; the remaining
        ``n - 1`` choices are deterministic given it.
    return_state : bool, default False
        Also return the running nearest-centre distance array, so a caller
        can continue the traversal without recomputing it.

    Returns
    -------
    numpy.ndarray or tuple
        ``(n,)`` int64 indices, or ``(indices, dist)`` when
        ``return_state=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from pros import farthest_first
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0]])
    >>> sorted(farthest_first(X, 2, seed_index=0).tolist())
    [0, 3]
    """
    if n <= 0:
        empty = np.empty(0, dtype=np.int64)
        return (empty, None) if return_state else empty
    state = FarthestFirst(X, seed_index=seed_index, rng=rng)
    state.run_to(n)
    idx = np.asarray(state.selected, dtype=np.int64)
    return (idx, state) if return_state else idx


def _pairwise_block_min(
    Q: np.ndarray,
    C: np.ndarray,
    c_sq: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centre distance and index for every row of ``Q``.

    Chunked over ``Q`` so peak memory is ``chunk * len(C)`` floats rather than
    ``len(Q) * len(C)``.
    """
    n_q = Q.shape[0]
    best_d2 = np.empty(n_q, dtype=np.float64)
    best_j = np.empty(n_q, dtype=np.int64)
    for start in range(0, n_q, chunk):
        stop = min(start + chunk, n_q)
        block = Q[start:stop]
        # (chunk, |C|) squared distances
        d2 = c_sq[None, :] - 2.0 * (block @ C.T)
        d2 += sq_norms(block)[:, None]
        np.maximum(d2, 0.0, out=d2)
        j = np.argmin(d2, axis=1)
        best_j[start:stop] = j
        best_d2[start:stop] = d2[np.arange(stop - start), j]
    return np.sqrt(best_d2), best_j


def assign_nearest(
    X: np.ndarray,
    centres: np.ndarray,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to, and index of, the nearest centre for each row of ``X``.

    Exact (brute force, chunked) nearest-centre assignment.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    C = np.ascontiguousarray(centres, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] == 0:
        raise ValueError("centres must be a non-empty 2-D array")
    if C.shape[1] != X.shape[1]:
        raise ValueError(
            f"dimension mismatch: X has {X.shape[1]}, centres have {C.shape[1]}"
        )
    return _pairwise_block_min(X, C, sq_norms(C), chunk)


def covering_radius(X: np.ndarray, centres: np.ndarray, chunk: int = 4096) -> float:
    """``max_i min_j d(x_i, c_j)`` -- the k-centre / minimax objective.

    This is the quantity the sketching literature reports as the (directed)
    Hausdorff distance from the full data to the sketch: the distance from
    the worst-covered data point to its nearest sketch member.  Smaller is
    better, and it is the metric PROS minimises.

    Parameters
    ----------
    X : numpy.ndarray
        ``(N, d)`` full data.
    centres : numpy.ndarray
        ``(n, d)`` sketch coordinates -- the points themselves, not indices.
    chunk : int, default 4096
        Rows of ``X`` processed per pass.  Bounds peak memory at
        ``chunk * n`` floats; it does not affect the result.

    Returns
    -------
    float
        The covering radius, in the units of ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> from pros import covering_radius
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0]])
    >>> covering_radius(X, X[[0, 1, 2]])   # point at 10 is 8 away
    8.0
    >>> covering_radius(X, X)              # sketch is the data
    0.0
    """
    d, _ = assign_nearest(X, centres, chunk=chunk)
    return float(d.max())


def min_pairwise_distance(P: np.ndarray, chunk: int = 4096) -> float:
    """Smallest distance between two distinct rows of ``P``.

    This is the maximin (separation) design objective, to be *maximised* by a
    good sketch: a large minimum pairwise distance means no two selected
    points are redundant with each other.  It is the complement of
    :func:`covering_radius`, which measures the opposite failure mode.

    Parameters
    ----------
    P : numpy.ndarray
        ``(n, d)`` point set, typically the sketch coordinates.
    chunk : int, default 4096
        Rows processed per pass; bounds peak memory without changing the
        result.

    Returns
    -------
    float
        The smallest pairwise distance.

    Examples
    --------
    >>> import numpy as np
    >>> from pros import min_pairwise_distance
    >>> min_pairwise_distance(np.array([[0.0], [1.0], [5.0]]))
    1.0
    """
    P = np.ascontiguousarray(P, dtype=np.float64)
    n = P.shape[0]
    if n < 2:
        return float("inf")
    p_sq = sq_norms(P)
    best = np.inf
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = P[start:stop]
        d2 = p_sq[None, :] - 2.0 * (block @ P.T)
        d2 += sq_norms(block)[:, None]
        # mask the diagonal
        rows = np.arange(stop - start)
        d2[rows, rows + start] = np.inf
        np.maximum(d2, 0.0, out=d2)
        m = d2.min()
        if m < best:
            best = m
    return float(np.sqrt(best))


def farthest_first_seeded(
    X: np.ndarray,
    n: int,
    seed_indices: np.ndarray,
    return_state: bool = False,
):
    """Farthest-first traversal *completing* a pre-selected set.

    ``seed_indices`` are taken as already selected; the traversal then adds
    points greedily until ``n`` are held in total.

    Notes
    -----
    The Gonzalez 2-approximation survives pre-seeding, but against a smaller
    reference.  Let ``k = len(seed_indices)`` and let ``R`` be the returned
    covering radius.  Every greedily added point was, at the moment of its
    selection, at distance ``>= R`` from all points already held (seeds
    included), because ``mindist`` only ever decreases.  Those ``n - k`` points
    together with the final witness of ``R`` are therefore pairwise ``>= R``
    apart, and a packing argument on that set gives

        ``R <= 2 * OPT_{n-k}(X)``.

    So mixing costs the difference between ``OPT_n`` and ``OPT_{n-k}`` rather
    than voiding the guarantee.  Since ``OPT_m ~ m^{-1/d}`` for data of
    intrinsic dimension ``d``, the predicted price of reserving a fraction
    ``beta`` of the budget is a factor ``(1 - beta)^{-1/d}`` — small at the
    ``d`` real single-cell data exhibit.
    """
    seed_indices = np.unique(np.asarray(seed_indices, dtype=np.int64))
    if seed_indices.size == 0:
        return farthest_first(X, n, return_state=return_state)
    state = FarthestFirst(X, seed_index=int(seed_indices[0]))
    for j in seed_indices[1:]:
        state.add(int(j))
    state.run_to(n)
    idx = np.asarray(state.selected, dtype=np.int64)
    return (idx, state) if return_state else idx
