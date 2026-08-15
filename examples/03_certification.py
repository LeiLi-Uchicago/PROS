"""Run PROS with a near-optimality certificate."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pros import sketch


path = pathlib.Path(__file__).resolve().parents[1] / "data" / "synthetic_clusters.npz"
X = np.load(path)["X"] if path.exists() else np.random.default_rng(0).random((900, 8))

res = sketch(X, n=60, r=4, partitioner="pc_tree", certify=True, seed=0)

print(f"radius:      {res.radius:.4f}")
print(f"rho:         {res.rho:.4f}")
print(f"opt lower:   {res.opt_lower:.4f}")
print(f"ratio upper: {res.ratio_upper:.3f}x")
