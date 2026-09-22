# PROS Reproduction Experiments

This folder contains standalone, reproducible parameter experiments for PROS.
It is not imported by the library or included in package distributions. All
generated data and results stay inside `test_para/`.

## Experiments

### 1. Gonzalez Initialization Sensitivity

Script: `test_gonzalez_init.py`

For each synthetic dataset, this fixes `n=100`, runs Gonzalez/FFT optimum
bounds for seeds `0..19`, and reports:

- `delta`
- `opt_lower = delta / 2`
- `opt_upper`
- `fft_radius_n`
- mean, standard deviation, CV, min/max/range, and max/min ratio

### 2. Partition and Allocation Comparison

Script: `compare_partition_allocation.py`

For each synthetic dataset and seed, this compares:

- `random + proportional`
- `kmeans + proportional`
- `kmeans + water_filling`

Each run records partition time, stage-1 time, stage-2 time, total runtime,
candidate-pool radius `rho`, and final covering radius.

This experiment requires `scikit-learn` because the k-means partitioner uses
`sklearn.cluster.MiniBatchKMeans`.

## Run

Run everything:

```bash
python3 test_para/run_all.py
```

Run each experiment separately:

```bash
python3 test_para/test_gonzalez_init.py
python3 test_para/compare_partition_allocation.py
```

If `scikit-learn` is missing, install project development dependencies:

```bash
python3 -m pip install -e ".[dev]"
```

## Outputs

Data:

- `data/balanced_clusters.npz`
- `data/imbalanced_clusters.npz`

Results:

- `results/gonzalez_init_runs.csv`
- `results/gonzalez_init_summary.csv`
- `results/gonzalez_init_summary.md`
- `results/partition_allocation_runs.csv`
- `results/partition_allocation_summary.csv`
- `results/partition_allocation_summary.md`
