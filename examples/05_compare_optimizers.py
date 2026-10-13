"""Compare FFT and upstream scSampler independently at both PROS stages.

Run from a checkout with its scsampler extra installed. See
docs/configuration.md for scSampler 1.0.2 / NumPy 2 compatibility notes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import pathlib
import platform
import sys
import time
from importlib.metadata import version

# Set these before importing numerical libraries for comparable timings.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pros import sketch  # noqa: E402
from pros.geometry import covering_radius  # noqa: E402
from pros.select import select_fft, select_scsampler  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=pathlib.Path, help="Coordinate .npy or .npz file"
    )
    parser.add_argument("--key", default="X", help="Coordinate key for .npz input")
    parser.add_argument("--n", type=int, default=60, help="Final sketch size")
    parser.add_argument("--r", type=float, default=4, help="Oversampling ratio")
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("optimizer_comparison.csv")
    )
    args = parser.parse_args()
    if args.data is None:
        X = np.random.default_rng(0).normal(size=(1200, 8))
    else:
        loaded = np.load(args.data, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                X = loaded[args.key]
        else:
            X = loaded

    versions = {"python_version": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn", "scsampler"):
        versions[package + "_version"] = version(package)
    # Import and initialize both paths before the measured runs. This uses a
    # separate RNG and does not alter the experiment's seeded stage-1 pools.
    warm = X[: min(64, len(X))]
    for select in (select_fft, select_scsampler):
        select(warm, min(5, len(warm)), np.random.default_rng(0))
    rows = []
    for seed in args.seeds:
        pools = {}
        combinations = list(itertools.product(("fft", "scsampler_maximin"), repeat=2))
        # Deterministically vary run order without consuming the PROS RNG.
        order = np.random.default_rng(seed).permutation(len(combinations))
        for position in order:
            selector, refiner = combinations[position]
            start = time.perf_counter()
            result = sketch(
                X,
                n=args.n,
                partitioner="random",
                n_blocks=args.n_blocks,
                allocator="proportional",
                r=args.r,
                selector=selector,
                refiner=refiner,
                seed=seed,
                mix=0,
                certify=False,
            )
            wall_s = time.perf_counter() - start
            # Refiners receive identical pools for each selector and seed.
            if selector in pools:
                np.testing.assert_array_equal(result.pool_indices, pools[selector])
            pools[selector] = result.pool_indices.copy()
            radius = covering_radius(X, X[result.indices])
            rho = covering_radius(X, X[result.pool_indices])
            row = dict(
                selector=selector,
                refiner=refiner,
                seed=seed,
                N=len(X),
                d=X.shape[1],
                n=args.n,
                r=args.r,
                partitioner="random",
                n_blocks=result.n_blocks,
                allocator="proportional",
                pool_size=len(result.pool_indices),
                wall_s=wall_s,
                stage1_s=result.timings["stage1"],
                stage2_s=result.timings["stage2"],
                radius=radius,
                pool_radius=rho,
                **versions,
            )
            rows.append(row)
            print(
                f"seed={seed} {selector} -> {refiner}: "
                f"radius={radius:.6f}, pool_radius={rho:.6f}, wall={wall_s:.3f}s"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
