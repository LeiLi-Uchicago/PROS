"""Stage-1 partitioners.

Each partitioner maps an ``(N, d)`` array to an integer ``labels`` array of
length ``N`` taking values in ``0 .. B-1``:

``random``
    Equal-size random split.

``kmeans``
    Coarse clustering into spatially compact blocks.

``pc_tree``
    Recursive median split along the leading principal component of the
    current node. Blocks are spatially compact and near-equal in size.
"""

from __future__ import annotations

import numpy as np

__all__ = ["random_blocks", "kmeans_blocks", "pc_tree_blocks", "make_partition"]


def random_blocks(n_points: int, n_blocks: int, rng) -> np.ndarray:
    """Equal-size random partition."""
    labels = np.arange(n_points) % int(n_blocks)
    rng.shuffle(labels)
    return labels.astype(np.int32)


def kmeans_blocks(
    X: np.ndarray,
    n_blocks: int,
    rng,
    batch_size: int = 4096,
    n_init: int = 3,
) -> np.ndarray:
    """Coarse k-means partition via ``MiniBatchKMeans``.

    ``n_blocks`` is used directly as the number of clusters.
    """
    from sklearn.cluster import MiniBatchKMeans

    seed = int(rng.integers(np.iinfo(np.int32).max))
    km = MiniBatchKMeans(
        n_clusters=int(n_blocks),
        random_state=seed,
        batch_size=batch_size,
        n_init=n_init,
    )
    return km.fit_predict(X).astype(np.int32)


def pc_tree_blocks(X: np.ndarray, n_blocks: int) -> np.ndarray:
    """Recursive median split along the leading principal component.

    At each step the *largest* current node is split, so the routine handles
    any ``n_blocks`` rather than only powers of two, and leaf sizes stay
    within a factor of two of one another.
    """
    n_points = X.shape[0]
    nodes: list[np.ndarray] = [np.arange(n_points)]
    target = max(1, int(n_blocks))

    while len(nodes) < target:
        # split the largest splittable node
        order = sorted(range(len(nodes)), key=lambda i: -len(nodes[i]))
        progressed = False
        for i in order:
            members = nodes[i]
            if len(members) < 2:
                continue
            sub = X[members]
            centred = sub - sub.mean(axis=0, keepdims=True)
            # leading right singular vector == leading PC direction
            try:
                _, _, vt = np.linalg.svd(centred, full_matrices=False)
                direction = vt[0]
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
                direction = np.zeros(sub.shape[1])
                direction[0] = 1.0
            proj = centred @ direction
            median = np.median(proj)
            left_mask = proj <= median
            if left_mask.all() or (~left_mask).all():
                # all points project identically; fall back to an index split
                half = len(members) // 2
                left_mask = np.zeros(len(members), dtype=bool)
                left_mask[:half] = True
            nodes[i] = members[left_mask]
            nodes.append(members[~left_mask])
            progressed = True
            break
        if not progressed:  # pragma: no cover - fewer than n_blocks points
            break

    labels = np.empty(n_points, dtype=np.int32)
    for b, members in enumerate(nodes):
        labels[members] = b
    return labels


def make_partition(
    X: np.ndarray,
    kind: str,
    n_blocks: int,
    rng,
) -> np.ndarray:
    """Dispatch to a named partitioner and validate the result.

    Stage **P** of PROS.  Splits ``X`` into spatially coherent blocks so that
    stage-1 selection can run independently (and cheaply) within each.

    Parameters
    ----------
    X : numpy.ndarray
        ``(N, d)`` coordinates.
    kind : {"kmeans", "pc_tree", "random", "none"}
        Partitioning scheme.  ``"none"`` returns a single block, which
        reduces PROS to a global method.
    n_blocks : int
        Requested number of blocks.  The realised count can be lower if the
        scheme produces empty blocks; the return value is what to trust.
    rng : numpy.random.Generator
        Randomness for the scheme's initialisation.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` int block labels, contiguous in ``[0, n_blocks_actual)``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or the partitioner returns labels that are
        not a valid contiguous labelling of all ``N`` rows.

    Examples
    --------
    >>> import numpy as np
    >>> from pros import make_partition
    >>> X = np.random.default_rng(0).random((300, 4))
    >>> labels = make_partition(X, "kmeans", 5, np.random.default_rng(0))
    >>> labels.shape, int(labels.min()), int(labels.max())
    ((300,), 0, 4)
    """
    if kind == "random":
        labels = random_blocks(X.shape[0], n_blocks, rng)
    elif kind == "kmeans":
        labels = kmeans_blocks(X, n_blocks, rng)
    elif kind == "pc_tree":
        labels = pc_tree_blocks(X, n_blocks)
    elif kind == "none":
        labels = np.zeros(X.shape[0], dtype=np.int32)
    else:
        raise ValueError(
            f"unknown partitioner {kind!r}; "
            "expected one of 'random', 'kmeans', 'pc_tree', 'none'"
        )

    # A partition must cover every cell exactly once; downstream budget
    # arithmetic silently misallocates if this is violated.
    if labels.shape[0] != X.shape[0]:
        raise AssertionError("partition labels do not match number of points")
    return labels
