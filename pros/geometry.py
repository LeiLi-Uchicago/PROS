"""Low-level geometric primitives for sketching.

The workhorse is :class:`FarthestFirst`, an *incremental* Gonzalez
farthest-first traversal (FFT).  Exposing the traversal as a resumable object
(rather than a one-shot function) is what lets the water-filling allocator in
:mod:`pros.allocate` reuse exactly the same code path as a plain
single-block FFT: it simply keeps one ``FarthestFirst`` per block in a priority
queue and advances whichever block currently has the largest covering radius.

Distances use SciPy's direct squared-Euclidean kernel to avoid catastrophic
cancellation in the norm/dot-product identity for translated coordinates.
Pairwise assignments are tiled across both points and centres.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ._validation import indices, integer, matrix

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
    The minimum separation of a prefix of k+1 points equals the radius
    after k points, and underpins the ``OPT >= delta / 2`` lower
    bound used by :mod:`pros.certify`.
    """

    def __init__(self, X: np.ndarray, seed_index: int | None = None, rng=None):
        self.X = matrix(X)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        self.n_points = self.X.shape[0]
        if self.n_points == 0:
            raise ValueError("X must contain at least one point")
        self.mindist2 = np.full(self.n_points, np.inf, dtype=np.float64)
        self.selected: list[int] = []
        self._selected_mask = np.zeros(self.n_points, dtype=bool)
        # Radius after each selection. radii[k] is the
        # covering radius achieved by the first k+1 selected points.
        self.radii: list[float] = []

        if seed_index is None:
            seed_index = int(rng.integers(self.n_points)) if rng is not None else 0
        self._add(integer(seed_index, "seed_index", maximum=self.n_points - 1))

    # -- internals ---------------------------------------------------------
    def _add(self, j: int) -> None:
        d2 = cdist(self.X, self.X[j : j + 1], metric="sqeuclidean")[:, 0]
        if not np.isfinite(d2).all():
            raise ValueError("distance overflow; rescale X")
        np.maximum(d2, 0.0, out=d2)  # guard against round-off negatives
        # A selected row is exactly covered by itself. The BLAS identity can
        # otherwise leave a tiny positive residual when every row is selected.
        d2[j] = 0.0
        np.minimum(self.mindist2, d2, out=self.mindist2)
        self.selected.append(int(j))
        self._selected_mask[j] = True
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
        j = int(np.argmax(np.where(self._selected_mask, -np.inf, self.mindist2)))
        hop_distance = float(np.sqrt(self.mindist2[j]))
        self._add(j)
        return j, hop_distance

    def add(self, j: int) -> None:
        """Force-select point ``j``, updating the covering state.

        Used to pre-seed a traversal (see :func:`farthest_first_seeded`).  No-op
        if ``j`` is already selected.
        """
        j = integer(j, "j", maximum=self.n_points - 1)
        if self._selected_mask[j]:
            return
        self._add(int(j))

    def run_to(self, n: int) -> list[int]:
        """Advance the traversal until ``n`` points are selected."""
        target = integer(n, "n", maximum=self.n_points)
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
        Also return the traversal state, so a caller
        can continue the traversal without recomputing it.

    Returns
    -------
    numpy.ndarray or tuple
        ``(n,)`` int64 indices, or ``(indices, FarthestFirst)`` when
        ``return_state=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from pros.geometry import farthest_first
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0]])
    >>> sorted(farthest_first(X, 2, seed_index=0).tolist())
    [0, 3]
    """
    X = matrix(X)
    n = integer(n, "n", maximum=len(X))
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return (empty, None) if return_state else empty
    state = FarthestFirst(X, seed_index=seed_index, rng=rng)
    state.run_to(n)
    idx = np.asarray(state.selected, dtype=np.int64)
    return (idx, state) if return_state else idx


def _pairwise_block_min(
    Q: np.ndarray,
    C: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centre distance and index for every row of ``Q``.

    Tiles both arrays so the distance buffer holds at most 1024 by 1024
    doubles, plus the linear-sized result arrays.
    """
    n_q = Q.shape[0]
    best_d2 = np.full(n_q, np.inf)
    best_j = np.zeros(n_q, dtype=np.int64)
    # A distance tile holds at most 1,048,576 doubles (~8 MiB).
    q_chunk = min(chunk, 1024)
    for start in range(0, n_q, q_chunk):
        stop = min(start + q_chunk, n_q)
        for cs in range(0, len(C), 1024):
            d2 = cdist(Q[start:stop], C[cs : cs + 1024], metric="sqeuclidean")
            if not np.isfinite(d2).all():
                raise ValueError("distance overflow; rescale coordinates")
            j = np.argmin(d2, axis=1)
            v = d2[np.arange(stop - start), j]
            better = v < best_d2[start:stop]
            best_d2[start:stop][better] = v[better]
            best_j[start:stop][better] = cs + j[better]
    return np.sqrt(best_d2), best_j


def assign_nearest(
    X: np.ndarray,
    centres: np.ndarray,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to, and index of, the nearest centre for each row of ``X``.

    Exact (brute force, chunked) nearest-centre assignment.
    """
    X = matrix(X)
    C = matrix(centres, "centres")
    chunk = integer(chunk, "chunk", minimum=1)
    if C.ndim != 2 or C.shape[0] == 0:
        raise ValueError("centres must be a non-empty 2-D array")
    if C.shape[1] != X.shape[1]:
        raise ValueError(
            f"dimension mismatch: X has {X.shape[1]}, centres have {C.shape[1]}"
        )
    return _pairwise_block_min(X, C, chunk)


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
        Maximum rows per pass (internally capped at 1024). Distance tiles
        use at most 1024 by 1024 doubles; it does not affect the result.

    Returns
    -------
    float
        The covering radius, in the units of ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> from pros.geometry import covering_radius
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
    >>> from pros.geometry import min_pairwise_distance
    >>> min_pairwise_distance(np.array([[0.0], [1.0], [5.0]]))
    1.0
    """
    P = matrix(P, "P")
    chunk = min(integer(chunk, "chunk", minimum=1), 1024)
    if len(P) < 2:
        return float("inf")
    best = np.inf
    for start in range(0, len(P), chunk):
        for cs in range(start, len(P), chunk):
            d2 = cdist(
                P[start : start + chunk], P[cs : cs + chunk], metric="sqeuclidean"
            )
            if not np.isfinite(d2).all():
                raise ValueError("distance overflow; rescale coordinates")
            if start == cs:
                np.fill_diagonal(d2, np.inf)
            best = min(best, float(d2.min()))
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
    X = matrix(X)
    n = integer(n, "n", maximum=len(X))
    seed_indices = indices(seed_indices, len(X), "seed_indices", allow_empty=True)
    if seed_indices.size > n:
        raise ValueError("seed count exceeds n")
    if seed_indices.size == 0:
        return farthest_first(X, n, return_state=return_state)
    state = FarthestFirst(X, seed_index=int(seed_indices[0]))
    for j in seed_indices[1:]:
        state.add(int(j))
    state.run_to(n)
    idx = np.asarray(state.selected, dtype=np.int64)
    return (idx, state) if return_state else idx
