# PROS

PROS (Partitioned and Refined Oversampling Sketches) selects a small,
diversity-preserving subset of rows from a large real-valued coordinate matrix.
It builds an oversampled candidate pool within partitions and performs one
global refinement pass over that pool. The returned subset is represented by
indices into the original matrix.

The package is domain-agnostic. It can be used with reduced single-cell
embeddings, as well as other feature matrices where Euclidean distance is a
meaningful geometry.

## Installation

PROS requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

Optional AnnData support:

```bash
python -m pip install -e ".[adata]"
```

For testing and linting:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from pros import sketch

X = np.random.default_rng(0).random((5_000, 50))
result = sketch(X, n=500, seed=0)

indices = result.indices
X_sketch = X[indices]
print(result.timings)
```

`result.indices` is a sorted, unique `int64` array of length `n`. By default,
PROS uses k-means partitioning, water-filling allocation, an oversampling ratio
of `r=10.0`, and global farthest-first refinement. Pass an explicit `seed` to
make a stochastic configuration reproducible.

## Certification

Set `certify=True` to compute the final covering radius, the candidate-pool
radius, and bounds derived from a full-data farthest-first traversal:

```python
result = sketch(X, n=500, seed=0, certify=True)

print(result.radius)       # R(X, S): final covering radius
print(result.rho)          # R(X, P): candidate-pool covering radius
print(result.opt_lower)    # lower bound on the optimal n-centre radius
print(result.ratio_upper)  # upper bound on R(X, S) / OPT_n(X)
```

Certification performs additional full-data work and is not included in
`result.timings["total"]`. Degenerate data may have `opt_lower == 0`; in that
case a finite approximation-ratio bound is not defined.

For an existing set of indices, use `pros.certificate`. `pros.opt_bounds`,
`pros.estimate_opt_scale`, and `pros.choose_r` are available for certificate
and oversampling-ratio workflows.

## AnnData

```python
from pros import sketch_adata

result = sketch_adata(adata, n=5_000, use_rep="X_pca", seed=0)
subset = adata[result.indices].copy()
```

`sketch_adata` also accepts an `.h5ad` path and reads it in backed mode before
extracting the requested representation.

## Configuration

```python
result = sketch(
    X,
    n=1_000,
    partitioner="kmeans",
    n_blocks="auto",
    allocator="water_filling",
    r=10.0,
    selector="fft",
    refiner="fft",
    seed=0,
)
```

The `partitioner`, within-partition `selector`, `allocator`, and global
`refiner` are modular. See [`docs/configuration.md`](docs/configuration.md) for
the supported values and [`docs/theory.md`](docs/theory.md) for the covering
objective and certificate definitions.

## Repository layout

- `pros/`: installable library code.
- `tests/`: automated package tests.
- `examples/`: small runnable examples using synthetic data in `data/`.
- `test_para/`: standalone parameter experiments and their generated outputs;
  this reproduction material is intentionally not packaged with the library.
- `docs/`: MkDocs source pages.

Run the examples from the repository root after installation:

```bash
python examples/01_basic_numpy.py
python examples/03_certification.py
```

## Citation

See [`CITATION.cff`](CITATION.cff) for software citation metadata.

## License

PROS is distributed under the [MIT License](LICENSE).
