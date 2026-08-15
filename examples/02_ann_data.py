"""Sketch an AnnData object on an embedding."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pros import sketch_adata

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover - example convenience
    raise SystemExit("Install AnnData with: python -m pip install -e '.[adata]'") from exc


rng = np.random.default_rng(0)
X_pca = rng.random((1000, 20))
adata = ad.AnnData(X=rng.poisson(1.0, size=(1000, 100)))
adata.obsm["X_pca"] = X_pca

res = sketch_adata(adata, n=100, use_rep="X_pca", partitioner="pc_tree", seed=0)
subset = adata[res.indices].copy()

print(subset)
print(f"stored sketch indices: {res.indices[:5]} ...")
