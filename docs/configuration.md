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
- `refiner`: `"fft"`, `"scsampler_maximin"`, `"maximin"`, `"local_swap"`, or `"none"`.
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

## Comparing FFT and scSampler

`scsampler_maximin` calls the upstream `scsampler.scsampler` function with
`random_split=1`. As a selector, it runs separately within each block. As a
refiner, it runs once on the pooled candidates and maps its selected positions
back to the original observations. It draws a reproducible upstream seed from
the PROS generator and restores NumPy's legacy RNG state after each call.

`maximin` is a separate PROS swap heuristic initialized by FFT; it does not
call upstream scSampler. `refine_max_iter` controls the swap heuristics only.

For the four optimizer combinations, hold partitioning, allocation, sketch
size, oversampling ratio, and seeds fixed:

```python
from itertools import product
from pros import sketch

results = {}
for selector, refiner in product(("fft", "scsampler_maximin"), repeat=2):
    results[selector, refiner] = sketch(
        X, n=1000, partitioner="random", n_blocks=32,
        allocator="proportional", r=10, selector=selector, refiner=refiner,
        seed=0, mix=0,
    )
```

Water-filling cannot be used for this comparison because it requires FFT
local selection. For a fixed selector and seed, both refiners receive the
same candidate pool. Certification remains optional and is separate from
sampling runtime; the FFT-specific theory bound is not asserted for scSampler
refinement.

Install the checkout and upstream dependencies with
`python -m pip install -e ".[scsampler]"`. This refiner is a local source change
and is not available in the published `pros-sketch==0.1.0` wheel.

Upstream scSampler 1.0.2 imports Scanpy but does not declare its dependencies;
the PROS extra includes Scanpy. With NumPy 2, upstream 1.0.2 also requires the
scalar-index compatibility patch used in the manuscript benchmarks: in
`scsampler/uclab.py`, replace `initial_index = np.random.randint(N, size=1)`
with `initial_index = int(np.random.randint(N, size=1)[0])`. Record this patch
with the environment. PROS does not patch installed dependencies or substitute
another algorithm when upstream fails.

`examples/05_compare_optimizers.py` runs all four combinations over three seeds
and writes full-data covering radius, candidate-pool radius, external sampling
wall time, stage timings, and dependency versions to CSV:

```bash
python examples/05_compare_optimizers.py --output optimizer_comparison.csv
python examples/05_compare_optimizers.py --data coordinates.npy \
    --n 1000 --r 10 --n-blocks 32 --seeds 0 1 2 --output comparison.csv
```

Metric evaluation is excluded from `wall_s`. On large datasets, full-data
radius evaluation can be expensive. The script warms up both optimizer paths
before timing and deterministically varies run order by seed. When writing
your own benchmark, also exclude first-call module import time from the
measured optimizer runs.

`mix>0` requires FFT refinement to preserve the uniform reserve. With
`mix_source="data"`, reserve points may enlarge the pool beyond the stage-1
`budget`; `m_per_block` still describes stage 1. `block_radii` and
`rho_blockwise_max` are available only with water-filling. An unfunded nonempty
block has infinite local covering radius.

The pool budget is `min(ceil(r*n), N)`. Larger r need not monotonically improve
the greedy final radius. `choose_r` reports `tolerance_met` and certifies only
its returned sketch. Its pool-radius evaluations can still be expensive.

