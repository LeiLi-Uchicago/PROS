from __future__ import annotations

import numpy as np

from pros.partition import make_partition


def test_partitioners_return_labels() -> None:
    X = np.random.default_rng(0).random((80, 4))

    for kind in ["random", "kmeans", "pc_tree", "none"]:
        labels = make_partition(X, kind, 4, np.random.default_rng(0))
        assert labels.shape == (80,)
        assert labels.min() == 0
        assert labels.max() < 4
