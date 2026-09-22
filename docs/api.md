# API

Public API:

- `pros.sketch(X, n, **kwargs)`: sketch a NumPy-compatible coordinate matrix.
- `pros.sketch_adata(adata, n, use_rep="X_pca", **kwargs)`: sketch an AnnData
  embedding or `.h5ad` path.
- `pros.certificate(X, sketch_indices, pool_indices=None)`: compute a
  certificate for an existing sketch.
- `pros.choose_r(X, n, tol=0.5)`: choose an oversampling ratio from a grid.
- `pros.estimate_opt_scale(X, n)`: estimate a target covering scale on a
  subsample.

Lower-level implementation modules are intentionally not re-exported from
`pros`. They may be useful for local experiments, but are not part of the
stable public API.
