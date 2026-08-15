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

Set `partitioner="none"` and `refiner="fft"` to run a plain global
farthest-first traversal.

