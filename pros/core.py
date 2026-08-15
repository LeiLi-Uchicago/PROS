"""The two-stage sketcher.

``sketch()`` is assembled from four swappable components so that every design
choice is an ablation axis rather than a hard-coded decision:

===============  =======================================================
component        options
===============  =======================================================
``partitioner``  ``random`` | ``kmeans`` | ``pc_tree`` | ``none``
``allocator``    ``proportional`` | ``power`` | ``volume`` | ``uniform``
                 | ``water_filling`` (fused with selection)
``selector``     ``fft`` | ``scsampler_maximin`` | ``random``
``refiner``      ``fft`` | ``maximin`` | ``local_swap`` | ``none``
===============  =======================================================

Reference points reachable by configuration:

* ``partitioner='random', allocator='proportional', selector='scsampler_maximin',
  r=1, refiner='none'``  -> scSampler-B<k>
* ``partitioner='pc_tree', allocator='water_filling', refiner='none'``
  -> Treehopper
* ``partitioner='none', refiner='fft'``  -> plain Hopper / Gonzalez FFT

so the ablation grid contains the published baselines as interior points and
any advantage of the proposed default is measured against them on identical
code paths.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from .allocate import make_allocation, water_filling
from .geometry import covering_radius
from .partition import make_partition
from .select import make_selector, refine

__all__ = ["sketch", "sketch_adata", "SketchResult"]


class SketchResult(dict):
    """Result of a :func:`sketch` call: a dict with attribute access.

    ``res.indices`` and ``res["indices"]`` are equivalent, so the object can
    be treated as a plain mapping (serialised, iterated, ``**``-expanded)
    while still reading like a record at the call site.

    Attributes
    ----------
    indices : numpy.ndarray
        ``(n,)`` sorted int64 indices into the input -- the sketch.
    pool_indices : numpy.ndarray
        Stage-1 candidate pool the refinement chose from.
    block_labels : numpy.ndarray
        ``(N,)`` block assignment from the partition stage.
    m_per_block : numpy.ndarray
        Pool quota allocated to each block.
    block_radii : numpy.ndarray
        Per-block covering radius of the stage-1 pool.
    rho_blockwise_max : float
        Largest block radius -- the blockwise bound on pool coarseness.
    n_blocks, budget, config : int, int, dict
        Realised block count, pool budget, and the resolved parameter set.
    timings : dict
        Seconds per stage: ``partition``, ``stage1``, ``stage2``, ``total``.
    radius, rho, opt_lower, opt_upper, ratio_upper, theory_bound, bound_slack
        Certificate fields; present only when ``sketch(..., certify=True)``.

    Examples
    --------
    >>> import numpy as np
    >>> from pros import sketch
    >>> res = sketch(np.random.default_rng(0).random((500, 5)), n=20)
    >>> res.indices is res["indices"]
    True
    >>> sorted(res.timings)
    ['partition', 'stage1', 'stage2', 'total']
    """

    __getattr__ = dict.__getitem__


#: Exponent and prefactor of the runtime-optimal block-count rule,
#: ``B* = C * N**BETA``, fitted to the runtime-optimal block count measured
#: at N in {10k, 50k, 200k, 800k} (results/param_sweep.csv, 360 runs).
#: Covering radius is nearly flat in block count over this grid -- the spread
#: across block counts is only ~3x the seed-to-seed SD -- so the block count
#: is chosen to minimise runtime, not quality.
AUTO_BLOCKS_BETA = 0.62
AUTO_BLOCKS_C = 0.0924
AUTO_BLOCKS_MIN = 8
AUTO_BLOCKS_MAX = 1024


def auto_n_blocks(n_cells: int) -> int:
    """Runtime-optimal block count for a population of ``n_cells`` points.

    Parameters
    ----------
    n_cells : int
        Population size ``N``.

    Returns
    -------
    int
        Block count, clipped to ``[8, 1024]``.  Never exceeds ``n_cells``.

    Notes
    -----
    The fixed default of 128 blocks was tuned at one population size.  It
    costs 2.6x runtime at ``N = 10,000`` (where 32 blocks is optimal) and
    1.6x at ``N = 800,000`` (where 512 is), for a covering radius that
    differs by less than one seed-to-seed standard deviation either way.

    Examples
    --------
    >>> auto_n_blocks(10_000)
    28
    >>> auto_n_blocks(800_000)
    422
    """
    b = AUTO_BLOCKS_C * float(n_cells) ** AUTO_BLOCKS_BETA
    return int(min(max(round(b), AUTO_BLOCKS_MIN), AUTO_BLOCKS_MAX, n_cells))


def sketch(
    X: np.ndarray,
    n: int,
    *,
    partitioner: str = "kmeans",
    n_blocks: int | str = "auto",
    allocator: str = "water_filling",
    r: float = 10.0,
    selector: str = "fft",
    refiner: str = "fft",
    d_intrinsic: float | None = None,
    alpha: float = 0.5,
    refine_max_iter: int = 25,
    mix: float = 0.0,
    mix_source: str = "pool",
    seed: int = 0,
    certify: bool = False,
    verbose: bool = False,
) -> SketchResult:
    """Select a diversity-preserving sketch of ``n`` points from ``X``.

    The generic entry point to PROS.  Runs the three stages the name
    describes -- **P**\\ artition into ``n_blocks``, **O**\\ versample a
    candidate pool by ``r``, **R**\\ efine globally over that pool -- and
    returns the resulting ``n`` indices.

    ``X`` is any real-valued coordinate matrix.  Nothing in the algorithm is
    specific to single-cell data; see :func:`sketch_adata` for the AnnData
    convenience wrapper.

    Parameters
    ----------
    X : numpy.ndarray
        ``(N, d)`` low-dimensional representation (e.g. 50 principal
        components), ideally min-max scaled to ``[0, 1]``.  Cast to
        C-contiguous ``float64`` internally.
    n : int
        Target sketch size, in ``(0, N]``.
    partitioner : {"kmeans", "grid", "none"}, default "kmeans"
        Stage-1 blocking scheme.  ``"none"`` collapses to a single block,
        which turns PROS into a plain global selection.
    n_blocks : int or "auto", default "auto"
        Number of stage-1 blocks.  More blocks means cheaper stage 1 and a
        coarser pool.  ``"auto"`` applies :func:`auto_n_blocks`, which scales
        the block count as ``N**0.62`` -- the runtime-optimal rule fitted in
        the parameter sweep.  Covering radius is nearly flat in this
        parameter, so the choice is a runtime decision; pass an int to
        override.
    allocator : {"water_filling", "proportional", "volume", "equal"}, default "water_filling"
        How the pool budget is divided among blocks.
    r : float, default 10.0
        Stage-1 oversampling ratio; the candidate pool holds ``min(r*n, N)``
        points.  ``r=1`` with ``refiner="none"`` recovers one-shot blockwise
        selection.  Quality saturates near ``r=10`` on the data tested.
    selector : {"fft", "random"}, default "fft"
        Within-block stage-1 selection rule.
    refiner : {"fft", "none"}, default "fft"
        Stage-2 global rule.  ``"none"`` disables refinement.
    d_intrinsic : float, optional
        Intrinsic dimension, required only by ``allocator="volume"``.
        Defaults to 5.0 with a warning if unset, since the ambient dimension
        is the wrong value to use there.
    alpha : float, default 0.5
        Water-filling exponent.
    refine_max_iter : int, default 25
        Cap on stage-2 refinement passes.
    mix : float, default 0.0
        Fraction of the sketch drawn uniformly at random instead of
        geometrically.  ``mix>0`` trades covering radius for proportional
        representation of abundant populations.
    mix_source : {"pool", "data"}, default "pool"
        Whether the uniform component is drawn from the candidate pool or the
        full data.
    seed : int, default 0
        Seed for the partitioner and any random draw.
    certify : bool, default False
        Compute the pool covering radius ``rho`` and the sketch covering
        radius, yielding a proven bound on distance from optimal.  This adds a
        full-data nearest-centre pass and is intended for evaluation; leave it
        ``False`` when timing the method.
    verbose : bool, default False
        Print per-stage progress.

    Returns
    -------
    SketchResult
        Dict-like, also attribute-accessible.  Always present: ``indices``
        (sorted, ``int64``), ``pool_indices``, ``block_labels``,
        ``m_per_block``, ``timings``.  When ``certify=True`` it also carries
        the certificate: ``radius``, ``rho``, ``opt_lower``, ``opt_upper``,
        ``ratio_upper`` (the certified factor from optimal), ``theory_bound``
        and ``bound_slack``.  See :func:`pros.certificate`.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, if ``n`` is outside ``(0, N]``, if ``r < 1``, or
        if ``mix_source`` is not one of the two accepted values.

    Notes
    -----
    Stage 2 is restricted to the pool, so cost scales with ``r*n`` rather than
    ``N``.  That is where the speed-up over a global farthest-first traversal
    comes from, and why ``r`` behaves as an accuracy dial: a denser pool can
    only improve what stage 2 has to choose from.

    Examples
    --------
    >>> import numpy as np
    >>> from pros import sketch
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((5000, 10))
    >>> res = sketch(X, n=200, seed=0)
    >>> res.indices.shape
    (200,)
    >>> len(np.unique(res.indices))
    200

    Ask for a certificate and read the proven bound:

    >>> res = sketch(X, n=200, seed=0, certify=True)
    >>> bool(res.ratio_upper >= 1.0)
    True

    Fewer, larger blocks with a denser pool:

    >>> res = sketch(X, n=200, n_blocks=16, r=20.0, seed=0)
    >>> res.indices.shape
    (200,)

    See Also
    --------
    sketch_adata : AnnData entry point.
    pros.certificate : recompute a certificate for an existing sketch.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.shape}")
    n_cells = X.shape[0]
    n = int(n)
    if not 0 < n <= n_cells:
        raise ValueError(f"n must be in (0, {n_cells}], got {n}")
    if r < 1:
        raise ValueError(f"oversampling ratio r must be >= 1, got {r}")
    if mix_source not in ("pool", "data"):
        raise ValueError(f'mix_source must be "pool" or "data", got {mix_source!r}')

    rng = np.random.default_rng(seed)
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    # ---- stage 0: partition -------------------------------------------
    t0 = time.perf_counter()
    if isinstance(n_blocks, str):
        if n_blocks != "auto":
            raise ValueError(
                f'n_blocks must be an int or "auto", got {n_blocks!r}'
            )
        resolved_blocks = auto_n_blocks(n_cells)
    else:
        resolved_blocks = int(n_blocks)
        if resolved_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {resolved_blocks}")
    effective_blocks = 1 if partitioner == "none" else resolved_blocks
    labels = make_partition(X, partitioner, effective_blocks, rng)
    n_blocks_actual = int(labels.max()) + 1
    timings["partition"] = time.perf_counter() - t0

    budget = int(min(np.ceil(r * n), n_cells))
    if budget < n:
        budget = n

    # ---- stage 1: per-block candidate selection ------------------------
    t0 = time.perf_counter()
    used_fallback = False
    if allocator == "water_filling":
        pool_indices, m_per_block, wf_info = water_filling(
            X, labels, n_blocks_actual, budget, rng
        )
    else:
        if allocator == "volume" and d_intrinsic is None:
            used_fallback = True
        m_per_block = make_allocation(
            X,
            labels,
            n_blocks_actual,
            budget,
            allocator,
            d_intrinsic=5.0 if d_intrinsic is None else float(d_intrinsic),
            alpha=alpha,
        )
        select_fn = make_selector(selector)
        chunks: list[np.ndarray] = []
        for b in range(n_blocks_actual):
            m_b = int(m_per_block[b])
            if m_b <= 0:
                continue
            members = np.flatnonzero(labels == b)
            if members.size == 0:
                continue
            local = select_fn(X[members], m_b, rng)
            chunks.append(members[np.asarray(local, dtype=np.int64)])
        pool_indices = (
            np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        )
        wf_info = {}
    pool_indices = np.unique(np.asarray(pool_indices, dtype=np.int64))
    timings["stage1"] = time.perf_counter() - t0

    if pool_indices.size < n:
        # Guarantee feasibility: top up the pool with random unselected cells.
        remaining = np.setdiff1d(np.arange(n_cells), pool_indices, assume_unique=False)
        need = n - pool_indices.size
        extra = rng.choice(remaining, size=min(need, remaining.size), replace=False)
        pool_indices = np.unique(np.concatenate([pool_indices, extra]))

    # ---- hybrid reserve ------------------------------------------------
    # ``mix`` reserves floor(mix*n) of the budget for a uniform draw.  Where
    # that draw comes from matters: refine() draws from the POOL, which is a
    # density-proportional sample of the data only when the pool has saturated
    # (r*n >= N).  At 68k cells with r=10 and n=10k the pool IS the whole
    # dataset, so pool-uniform == data-uniform; at 1.29M it is a diversified
    # 10% subset that already over-represents sparse regions, and a pool-uniform
    # reserve inherits that bias.  ``mix_source="data"`` draws uniformly over
    # all N cells and adds the draw to the candidate set, so the reserve is
    # genuinely density-proportional at any N.  Proposition 2 applies either
    # way -- it assumes nothing about where the seeds come from.
    mix_seeds = None
    if mix > 0.0 and mix_source == "data":
        k = int(round(mix * n))
        if k > 0:
            reserve = rng.choice(n_cells, size=min(k, n_cells), replace=False)
            pool_indices = np.union1d(pool_indices, reserve.astype(np.int64))
            mix_seeds = np.searchsorted(pool_indices, np.unique(reserve))
            if not np.array_equal(pool_indices[mix_seeds], np.unique(reserve)):
                raise AssertionError("reserve positions misaligned with pool")

    # ---- stage 2: global refinement ------------------------------------
    t0 = time.perf_counter()
    local_final = refine(
        X[pool_indices], n, refiner, rng, max_iter=refine_max_iter, mix=mix,
        mix_seeds=mix_seeds,
    )
    final = np.sort(pool_indices[np.asarray(local_final, dtype=np.int64)])
    timings["stage2"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_start

    if final.size != n:
        raise AssertionError(f"sketch returned {final.size} cells, expected {n}")
    if np.unique(final).size != n:
        raise AssertionError("sketch contains duplicate indices")

    result = SketchResult(
        indices=final,
        pool_indices=pool_indices,
        block_labels=labels,
        m_per_block=np.asarray(m_per_block, dtype=np.int64),
        n_blocks=n_blocks_actual,
        budget=budget,
        timings=timings,
        config=dict(
            partitioner=partitioner,
            # What the caller asked for ("auto" or an int) and what that
            # resolved to, so a run recorded with n_blocks="auto" is still
            # exactly reproducible after the auto rule changes.
            n_blocks=n_blocks,
            n_blocks_resolved=resolved_blocks,
            allocator=allocator,
            r=r,
            selector=selector,
            refiner=refiner,
            seed=seed,
        ),
        volume_allocator_missing_dim=used_fallback,
        **wf_info,
    )

    if certify:
        # Produce the FULL certificate, not just the two radii.  The reported
        # headline quantity is ``ratio_upper`` (the certified factor from
        # optimal); a user who passes certify=True is asking for exactly that,
        # and previously had to call pros.certificate() separately to get
        # it.  This costs one extra Gonzalez pass for the optimum bounds, which
        # is why certification is opt-in and is excluded from reported timings.
        t0 = time.perf_counter()
        from .certify import certificate

        cert = certificate(X, final, pool_indices=pool_indices, seed=seed)
        cert_timings = cert.pop("timings", {})
        result.update(cert)
        timings["certify"] = time.perf_counter() - t0
        timings.update({f"certify_{k}": v for k, v in cert_timings.items()})

    if verbose:
        print(
            f"n={n} blocks={n_blocks_actual} pool={pool_indices.size} "
            f"total={timings['total']:.2f}s"
        )
    return result


def sketch_adata(
    adata,
    n: int,
    use_rep: str = "X_pca",
    *,
    return_adata: bool = False,
    copy: bool = True,
    **kwargs,
):
    """Sketch an :class:`anndata.AnnData` on one of its ``obsm`` embeddings.

    The AnnData entry point to PROS.  It resolves the embedding, delegates to
    :func:`sketch`, and optionally returns the subset object rather than the
    indices.

    Parameters
    ----------
    adata : anndata.AnnData or str or pathlib.Path
        An AnnData object, or a path to an ``.h5ad`` file.  A path is read
        with ``backed='r'`` so that only the embedding is pulled into memory;
        this makes million-cell files sketchable without loading the counts.
    n : int
        Target sketch size, in ``(0, adata.n_obs]``.
    use_rep : str, default ``"X_pca"``
        Key in ``adata.obsm`` holding the coordinates to sketch on.  Use
        ``"X"`` to sketch on ``adata.X`` directly (dense only, and rarely what
        you want -- PROS is a geometric method and expects a reduced space).
    return_adata : bool, default False
        If True, return the subset AnnData instead of the
        :class:`SketchResult`.  The result object is attached to the subset as
        ``.uns["pros"]`` so the certificate is not lost.
    copy : bool, default True
        Only consulted when ``return_adata`` is True.  If False, and the input
        is backed, the returned view is not materialised.
    **kwargs
        Forwarded verbatim to :func:`sketch` (``n_blocks``, ``r``, ``seed``,
        ``certify``, ...).

    Returns
    -------
    SketchResult or anndata.AnnData
        Depending on ``return_adata``.

    Raises
    ------
    KeyError
        If ``use_rep`` is absent from ``adata.obsm``.  The message lists the
        keys that *are* present, because a wrong embedding name is the most
        common call-site error.

    Examples
    --------
    >>> import scanpy as sc                                 # doctest: +SKIP
    >>> adata = sc.read_h5ad("pbmc.h5ad")                   # doctest: +SKIP
    >>> res = sketch_adata(adata, n=5000, certify=True)     # doctest: +SKIP
    >>> sub = adata[res.indices]                            # doctest: +SKIP

    Sketch straight from disk without loading the matrix:

    >>> res = sketch_adata("pbmc.h5ad", n=5000)             # doctest: +SKIP

    See Also
    --------
    sketch : the generic matrix entry point.
    """
    opened = None
    if isinstance(adata, (str, pathlib.Path)):
        import anndata as ad

        opened = ad.read_h5ad(str(adata), backed="r")
        adata = opened

    if use_rep == "X":
        X = adata.X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    else:
        if use_rep not in adata.obsm:
            raise KeyError(
                f"{use_rep!r} not in adata.obsm (available: {list(adata.obsm)})"
            )
        X = np.asarray(adata.obsm[use_rep])

    res = sketch(X, n, **kwargs)

    if not return_adata:
        return res

    sub = adata[res.indices]
    if copy:
        sub = sub.to_memory() if hasattr(sub, "to_memory") else sub.copy()
    # Keep the certificate with the data it describes.
    sub.uns["pros"] = {k: v for k, v in res.items() if k != "block_labels"}
    return sub
