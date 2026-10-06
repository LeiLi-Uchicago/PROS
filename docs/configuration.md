# Configuration

The default configuration is intended to be a strong starting point:

```python
res = sketch(
    X,
    n=1000,
    partitioner="kmeans",
    n_blocks="auto",
    allocator="water_filling",
    r=10.0,
    selector="fft",
    refiner="fft",
    seed=0,
)
```

Useful options:

- `partitioner`: `"kmeans"`, `"pc_tree"`, `"random"`, or `"none"`.
- `allocator`: `"water_filling"`, `"proportional"`, `"power"`, `"volume"`, or
  `"uniform"`.
- `selector`: `"fft"`, `"scsampler_maximin"`, or `"random"`.
- `refiner`: `"fft"`, `"maximin"`, `"local_swap"`, or `"none"`.
- `r`: oversampling ratio. Larger values make a denser pool and cost more.
- `certify`: compute certificate fields for evaluation.

For a single global FFT selection, use `partitioner="none", r=1,
refiner="none"`: stage 1 selects n points and stage 2 retains the whole pool.
With r>1, the single-block configuration still runs two selection stages.

Water-filling requires `selector="fft"`; incompatible combinations raise an
error. Other allocators support the other selectors. scSampler failures are
propagated, never replaced silently with FFT. Fixed seeds are reproducible
within a fixed numerical/software environment; record dependency versions and
thread settings for experiments.

`mix>0` requires FFT refinement to preserve the uniform reserve. With
`mix_source="data"`, reserve points may enlarge the pool beyond the stage-1
`budget`; `m_per_block` still describes stage 1. `block_radii` and
`rho_blockwise_max` are available only with water-filling. An unfunded nonempty
block has infinite local covering radius.

The pool budget is `min(ceil(r*n), N)`. Larger r need not monotonically improve
the greedy final radius. `choose_r` reports `tolerance_met` and certifies only
its returned sketch. Its pool-radius evaluations can still be expensive.

