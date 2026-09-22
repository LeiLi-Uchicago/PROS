"""Synthetic datasets for PROS parameter experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


DATASETS = (
    DatasetSpec("balanced_clusters", DATA_DIR / "balanced_clusters.npz"),
    DatasetSpec("imbalanced_clusters", DATA_DIR / "imbalanced_clusters.npz"),
)


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def _balanced_clusters(seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_clusters = 6
    n_per_cluster = 500
    dim = 10
    centres = rng.normal(0.0, 4.0, size=(n_clusters, dim))
    parts = []
    labels = []
    for cluster_id, centre in enumerate(centres):
        cov_scale = 0.45 + 0.08 * (cluster_id % 3)
        part = centre + rng.normal(0.0, cov_scale, size=(n_per_cluster, dim))
        parts.append(part)
        labels.append(np.full(n_per_cluster, cluster_id, dtype=np.int64))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    order = rng.permutation(X.shape[0])
    return _standardize(X[order]), y[order]


def _imbalanced_clusters(seed: int = 456) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = 10
    sizes = np.array([1700, 700, 350, 160, 70, 20])
    scales = np.array([0.22, 0.35, 0.75, 1.05, 0.55, 1.35])
    centres = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.2, 0.2, -0.1, 0.4, 0.1, -0.2, 0.0, 0.3, -0.1, 0.2],
            [-2.5, 2.9, 0.7, -0.8, 0.3, 0.2, -0.4, 0.1, 0.6, -0.2],
            [1.8, -3.4, 2.2, 0.5, -0.6, 0.8, 0.1, -0.7, 0.3, 0.4],
            [-4.0, -2.0, -1.8, 1.6, 1.0, -0.5, 0.9, 0.2, -0.3, 0.7],
            [5.0, 4.6, -3.8, 2.8, -2.2, 1.5, -1.4, 1.2, 0.8, -0.9],
        ],
        dtype=np.float64,
    )
    parts = []
    labels = []
    for cluster_id, (size, scale, centre) in enumerate(zip(sizes, scales, centres)):
        part = centre + rng.normal(0.0, float(scale), size=(int(size), dim))
        parts.append(part)
        labels.append(np.full(int(size), cluster_id, dtype=np.int64))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    order = rng.permutation(X.shape[0])
    return _standardize(X[order]), y[order]


def ensure_datasets(force: bool = False) -> list[DatasetSpec]:
    """Create cached synthetic datasets if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generators = {
        "balanced_clusters": _balanced_clusters,
        "imbalanced_clusters": _imbalanced_clusters,
    }
    for spec in DATASETS:
        if spec.path.exists() and not force:
            continue
        X, labels = generators[spec.name]()
        np.savez_compressed(spec.path, X=X, labels=labels)
    return list(DATASETS)


def load_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load one cached synthetic dataset."""
    ensure_datasets()
    with np.load(spec.path) as data:
        return np.asarray(data["X"], dtype=np.float64), np.asarray(data["labels"])


if __name__ == "__main__":
    for dataset in ensure_datasets(force=True):
        X, labels = load_dataset(dataset)
        print(f"{dataset.name}: X={X.shape}, labels={labels.shape}, path={dataset.path}")

