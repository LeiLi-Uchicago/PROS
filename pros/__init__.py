"""PROS: Partitioned and Refined Oversampling Sketches.

The public API is intentionally small. Use :func:`sketch` for coordinate
matrices, :func:`sketch_adata` for AnnData, and the certificate helpers to
evaluate a completed sketch.
"""

from ._options import options
from .certify import certificate, choose_r, estimate_opt_scale, opt_bounds
from .core import SketchResult, sketch, sketch_adata

__version__ = "0.1.1"

__all__ = [
    "sketch",
    "sketch_adata",
    "SketchResult",
    "certificate",
    "opt_bounds",
    "choose_r",
    "estimate_opt_scale",
    "options",
    "__version__",
]
