# PROS

PROS is a Python package for **P**artitioned and **R**efined **O**versampling
**S**ketches: fast, diversity-preserving geometric sketches of large coordinate
matrices.

The package was designed for reduced single-cell embeddings such as PCA
coordinates, but the core algorithm works on any real-valued `numpy` matrix.
It returns representative row indices and, when requested, a run-time
certificate for the k-center covering objective.

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

With optional AnnData support:

```bash
python -m pip install -e ".[adata]"
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from pros import sketch

rng = np.random.default_rng(0)
X = rng.random((5000, 50))

res = sketch(X, n=500, seed=0)
print(res.indices.shape)
print(res.timings)
```

`res.indices` is an integer array into the original matrix `X`.

## Certified Runs

Certification computes the sketch covering radius, the pool covering radius,
and a proven upper bound on the approximation ratio. It is useful for
evaluation and reporting, but it performs extra full-data passes and should not
be included in fast-path timing.

```python
res = sketch(X, n=500, certify=True, seed=0)

print(res.radius)
print(res.rho)
print(res.ratio_upper)
```

## AnnData

```python
from pros import sketch_adata

res = sketch_adata(adata, n=5000, use_rep="X_pca", certify=True)
subset = adata[res.indices]
```

`sketch_adata` also accepts a path to an `.h5ad` file and reads it in backed
mode so the embedding can be sketched without loading the full count matrix.

## Configuration

The main entry point is:

```python
from pros import sketch

res = sketch(
    X,
    n=1000,
    partitioner="kmeans",
    allocator="water_filling",
    r=10.0,
    selector="fft",
    refiner="fft",
    seed=0,
)
```

Common knobs:

- `n`: final sketch size.
- `r`: oversampling ratio; the stage-1 pool has up to `r * n` candidates.
- `n_blocks`: `"auto"` by default, or an explicit block count.
- `partitioner`: `"kmeans"`, `"pc_tree"`, `"random"`, or `"none"`.
- `allocator`: `"water_filling"`, `"proportional"`, `"power"`, `"volume"`, or
  `"uniform"`.
- `certify`: add a near-optimality certificate to the result.

## Examples

Runnable examples live in `examples/`:

```bash
python examples/01_basic_numpy.py
python examples/03_certification.py
```

The tiny dataset in `data/synthetic_clusters.npz` is synthetic and intended
only for documentation, tests, and smoke runs.

## License

MIT License. See `LICENSE`.

