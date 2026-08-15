# API

Primary entry points:

- `pros.sketch(X, n, **kwargs)`: sketch a NumPy-compatible coordinate matrix.
- `pros.sketch_adata(adata, n, use_rep="X_pca", **kwargs)`: sketch an AnnData
  embedding or `.h5ad` path.
- `pros.certificate(X, sketch_indices, pool_indices=None)`: compute a
  certificate for an existing sketch.
- `pros.choose_r(X, n, tol=0.5)`: choose an oversampling ratio from a grid.

Lower-level components are also public for experiments and ablations:

- `pros.make_partition`
- `pros.make_allocation`
- `pros.water_filling`
- `pros.farthest_first`
- `pros.covering_radius`
- `pros.min_pairwise_distance`

