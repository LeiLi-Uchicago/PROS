"""Compare a few PROS configurations on the bundled synthetic data."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pros import sketch


path = pathlib.Path(__file__).resolve().parents[1] / "data" / "synthetic_clusters.npz"
X = np.load(path)["X"] if path.exists() else np.random.default_rng(0).random((900, 8))

configs = [
    {"partitioner": "pc_tree", "allocator": "water_filling"},
    {"partitioner": "random", "allocator": "proportional"},
    {"partitioner": "none", "allocator": "uniform"},
]

for cfg in configs:
    res = sketch(X, n=60, r=4, certify=True, seed=0, **cfg)
    name = ", ".join(f"{k}={v}" for k, v in cfg.items())
    print(f"{name:52s} radius={res.radius:.4f} ratio<={res.ratio_upper:.3f}x")
