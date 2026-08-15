"""PROS -- Partitioned and Refined Oversampling Sketch.

Certified geometric sketching of large datasets.  The name states the
algorithm:

**P**\\ artition
    Split the data into ``n_blocks`` spatially coherent blocks.
**O**\\ versample
    Draw ``r`` times more candidates per block than the block's final quota,
    building a pool that is cheap to construct yet provably dense.
**R**\\ efine
    Run one global selection pass restricted to that pool, recovering the
    quality of a global method at a fraction of its cost.
**S**\\ ketch
    The result: ``n`` representative points, plus a certificate.

The pool's covering radius ``rho`` is measurable at run time, which turns the
oversampling ratio into a requested accuracy tolerance and lets each run emit a
near-optimality certificate -- a proven upper bound on how far the returned
sketch is from the optimal one.

Although developed on single-cell data, PROS operates on any real-valued
coordinate matrix and carries no domain-specific assumptions.

Quick start
-----------
>>> import numpy as np
>>> from pros import sketch
>>> X = np.random.default_rng(0).random((5000, 50))   # e.g. 50 PCs
>>> res = sketch(X, n=500)
>>> res.indices.shape
(500,)

``res.indices`` is an int64 index array into ``X``.  For AnnData input use
:func:`pros.sketch_adata`, and pass ``certify=True`` to either entry point to
attach a proven near-optimality bound to the run.
"""

from .allocate import allocate_volume, make_allocation, water_filling
from .certify import certificate, choose_r, opt_bounds
from .core import SketchResult, auto_n_blocks, sketch, sketch_adata
from .geometry import covering_radius, farthest_first, min_pairwise_distance
from .partition import make_partition

__version__ = "0.1.0"

__all__ = [
    "sketch",
    "sketch_adata",
    "SketchResult",
    "auto_n_blocks",
    "certificate",
    "opt_bounds",
    "choose_r",
    "covering_radius",
    "farthest_first",
    "min_pairwise_distance",
    "make_partition",
    "make_allocation",
    "allocate_volume",
    "water_filling",
    "__version__",
]
