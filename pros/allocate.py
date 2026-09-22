"""Stage-1 candidate-budget allocators.

An allocator divides a pool budget among blocks. ``water_filling`` combines
allocation with incremental farthest-first selection, prioritising the block
with the largest current covering radius.
"""

from __future__ import annotations

import heapq

import numpy as np

from .geometry import FarthestFirst

__all__ = [
    "round_to_budget",
    "allocate_proportional",
    "allocate_power",
    "allocate_volume",
    "water_filling",
    "make_allocation",
]


def round_to_budget(
    weights: np.ndarray,
    budget: int,
    caps: np.ndarray,
    floors: np.ndarray | None = None,
) -> np.ndarray:
    """Turn real-valued ``weights`` into integers summing to ``budget``.

    Respects per-block ``caps`` (a block cannot contribute more candidates than
    it has cells) and ``floors`` (default 1, so no block is ever fully
    silenced).  Uses proportional apportionment with cap redistribution,
    falling back to largest-remainder for the final few units.
    """
    w = np.asarray(weights, dtype=np.float64)
    caps = np.asarray(caps, dtype=np.int64)
    n_blocks = w.shape[0]
    budget = int(min(int(budget), int(caps.sum())))
    if budget <= 0:
        return np.zeros(n_blocks, dtype=np.int64)

    floors = (
        np.ones(n_blocks, dtype=np.int64)
        if floors is None
        else np.asarray(floors, dtype=np.int64)
    )
    floors = np.minimum(floors, caps)

    # Budget too small to fund every floor: fund the highest-weight blocks.
    if floors.sum() > budget:
        m = np.zeros(n_blocks, dtype=np.int64)
        remaining = budget
        for i in np.argsort(-w):
            if remaining <= 0:
                break
            take = int(min(floors[i], remaining))
            m[i] = take
            remaining -= take
        return m

    m = floors.copy()
    remaining = budget - int(m.sum())

    while remaining > 0:
        active = m < caps
        if not active.any():
            break
        w_active = np.where(active, np.maximum(w, 0.0), 0.0)
        total = w_active.sum()

        if total <= 0:  # zero weights: spread evenly over active blocks
            for i in np.flatnonzero(active):
                if remaining <= 0:
                    break
                m[i] += 1
                remaining -= 1
            continue

        share = w_active / total * remaining
        add = np.minimum(np.floor(share).astype(np.int64), caps - m)
        if add.sum() == 0:  # largest remainder for the tail
            frac = np.where(active & (m < caps), share - np.floor(share), -1.0)
            for i in np.argsort(-frac):
                if remaining <= 0:
                    break
                if m[i] < caps[i]:
                    m[i] += 1
                    remaining -= 1
            continue
        m += add
        remaining = budget - int(m.sum())

    return m


def _block_sizes(labels: np.ndarray, n_blocks: int) -> np.ndarray:
    return np.bincount(labels, minlength=n_blocks).astype(np.int64)


def allocate_proportional(labels: np.ndarray, n_blocks: int, budget: int) -> np.ndarray:
    """Allocate candidates in proportion to block size."""
    sizes = _block_sizes(labels, n_blocks)
    return round_to_budget(sizes.astype(np.float64), budget, sizes)


def allocate_power(
    labels: np.ndarray, n_blocks: int, budget: int, alpha: float = 0.5
) -> np.ndarray:
    """``m_b ~ N_b^alpha`` -- interpolates proportional (1) and uniform (0).

    This heuristic up-weights small blocks relative to proportional allocation.
    """
    sizes = _block_sizes(labels, n_blocks)
    w = np.power(sizes.astype(np.float64), float(alpha))
    return round_to_budget(w, budget, sizes)


def block_radius_scale(
    X: np.ndarray,
    labels: np.ndarray,
    n_blocks: int,
    quantile: float = 0.75,
) -> np.ndarray:
    """Characteristic linear extent of each block.

    Uses a high quantile of the distance to the block centroid rather than the
    maximum, so a single outlying cell cannot inflate a block's volume
    estimate.  Empty blocks get scale 0.
    """
    scales = np.zeros(n_blocks, dtype=np.float64)
    for b in range(n_blocks):
        members = np.flatnonzero(labels == b)
        if members.size == 0:
            continue
        sub = X[members]
        centroid = sub.mean(axis=0)
        d = np.sqrt(np.maximum(((sub - centroid) ** 2).sum(axis=1), 0.0))
        scales[b] = float(np.quantile(d, quantile)) if members.size > 1 else 0.0
    return scales


def allocate_volume(
    X: np.ndarray,
    labels: np.ndarray,
    n_blocks: int,
    budget: int,
    d_intrinsic: float,
    quantile: float = 0.75,
) -> np.ndarray:
    """``m_b ~ V_b ~ scale_b ^ d`` -- closed-form water-filling.

    ``d_intrinsic`` should be an *intrinsic* dimension estimate (e.g. TwoNN),
    not the ambient PC count: using ambient ``d`` makes the exponent wildly too
    large and collapses the budget onto a single block.
    """
    sizes = _block_sizes(labels, n_blocks)
    scales = block_radius_scale(X, labels, n_blocks, quantile=quantile)
    d = float(max(d_intrinsic, 1.0))
    # Normalise before exponentiating to keep scale_b^d in a sane numeric range.
    s_max = scales.max()
    if s_max <= 0:
        return allocate_proportional(labels, n_blocks, budget)
    w = np.power(scales / s_max, d)
    return round_to_budget(w, budget, sizes)


def water_filling(
    X: np.ndarray,
    labels: np.ndarray,
    n_blocks: int,
    budget: int,
    rng,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exact greedy water-filling, fusing allocation with selection.

    Maintains one incremental :class:`FarthestFirst` per block in a max-heap
    keyed on current covering radius, and repeatedly advances whichever block
    is currently worst-covered.  This greedily minimises ``max_b R_b = rho``
    and needs no dimension estimate or volume proxy.

    Returns ``(pool_indices, m_per_block, info)``.
    """
    sizes = _block_sizes(labels, n_blocks)
    budget = int(min(int(budget), int(sizes.sum())))
    members = [np.flatnonzero(labels == b) for b in range(n_blocks)]

    states: dict[int, FarthestFirst] = {}
    pool: list[int] = []
    heap: list[tuple[float, int, int]] = []
    counts = np.zeros(n_blocks, dtype=np.int64)

    # Seed one candidate per non-empty block, then let radius drive the rest.
    for b in range(n_blocks):
        if members[b].size == 0 or len(pool) >= budget:
            continue
        st = FarthestFirst(X[members[b]], rng=rng)
        states[b] = st
        pool.append(int(members[b][st.selected[0]]))
        counts[b] += 1
        if not st.exhausted:
            # negate radius for a max-heap via heapq's min-heap
            heapq.heappush(heap, (-st.radius, b, 0))

    tie = 0
    while len(pool) < budget and heap:
        neg_radius, b, _ = heapq.heappop(heap)
        st = states[b]
        j, _ = st.hop()
        pool.append(int(members[b][j]))
        counts[b] += 1
        if not st.exhausted:
            tie += 1
            heapq.heappush(heap, (-st.radius, b, tie))

    final_radii = np.array(
        [states[b].radius if b in states else 0.0 for b in range(n_blocks)],
        dtype=np.float64,
    )
    info = {
        "block_radii": final_radii,
        "rho_blockwise_max": float(final_radii.max()) if n_blocks else 0.0,
    }
    return np.asarray(pool, dtype=np.int64), counts, info


def make_allocation(
    X: np.ndarray,
    labels: np.ndarray,
    n_blocks: int,
    budget: int,
    kind: str,
    d_intrinsic: float = 5.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """Dispatch to a named allocator, returning the per-block quota ``m_b``.

    Stage **O** of PROS decides how many candidates each block contributes to
    the oversampled pool.  A block covering a large, sparse region needs more
    candidates than a small dense one; that is what the allocator encodes.

    Parameters
    ----------
    X : numpy.ndarray
        ``(N, d)`` coordinates.
    labels : numpy.ndarray
        ``(N,)`` block labels from :func:`make_partition`.
    n_blocks : int
        Number of blocks.
    budget : int
        Total pool size to distribute, i.e. ``min(r * n, N)``.
    kind : {"proportional", "power", "volume", "uniform"}
        Allocation rule.  ``"proportional"`` follows block population,
        ``"power"`` follows block population raised to ``alpha``, ``"volume"``
        follows estimated block volume (and needs ``d_intrinsic``), and
        ``"uniform"`` spreads budget as evenly as caps allow.  The fused
        ``"water_filling"`` allocator is handled by :func:`pros.sketch`
        because it performs allocation and candidate selection together.
    d_intrinsic : float, default 5.0
        Intrinsic dimension estimate, used only by ``"volume"``.
    alpha : float, default 0.5
        Exponent for ``"power"`` allocation: 1 is population-proportional and
        0 is uniform over non-empty blocks.

    Returns
    -------
    numpy.ndarray
        ``(n_blocks,)`` int quotas summing to ``budget``.

    Examples
    --------
    >>> import numpy as np
    >>> from pros import make_allocation
    >>> X = np.random.default_rng(0).random((300, 3))
    >>> labels = np.repeat([0, 1, 2], 100)
    >>> m = make_allocation(X, labels, 3, 60, "proportional")
    >>> int(m.sum())
    60
    """
    if kind == "proportional":
        return allocate_proportional(labels, n_blocks, budget)
    if kind == "power":
        return allocate_power(labels, n_blocks, budget, alpha=alpha)
    if kind == "volume":
        return allocate_volume(X, labels, n_blocks, budget, d_intrinsic=d_intrinsic)
    if kind == "uniform":
        sizes = _block_sizes(labels, n_blocks)
        return round_to_budget(np.ones(n_blocks), budget, sizes)
    raise ValueError(
        f"unknown allocator {kind!r}; expected 'proportional', 'power', "
        "'volume', 'uniform', or the fused 'water_filling'"
    )
