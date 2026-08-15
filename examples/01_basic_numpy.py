"""Basic PROS sketching on a NumPy matrix."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pros import sketch


def load_example_matrix() -> np.ndarray:
    path = pathlib.Path(__file__).resolve().parents[1] / "data" / "synthetic_clusters.npz"
    if path.exists():
        return np.load(path)["X"]
    return np.random.default_rng(0).random((900, 8))


X = load_example_matrix()
res = sketch(X, n=60, r=4, partitioner="pc_tree", seed=0)

print(f"selected {res.indices.size} rows from {X.shape[0]}")
print(f"pool size: {res.pool_indices.size}")
print(f"timings: {res.timings}")
