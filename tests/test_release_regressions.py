import copy
import itertools
import sys
import types

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from pros import SketchResult, certificate, choose_r, opt_bounds, sketch, sketch_adata
from pros.allocate import round_to_budget, water_filling
from pros.geometry import (
    assign_nearest,
    covering_radius,
    farthest_first,
    farthest_first_seeded,
)
from pros.partition import pc_tree_blocks
from pros.select import kcenter_local_search, select_scsampler


@pytest.mark.parametrize("partitioner", ["random", "none", "pc_tree", "kmeans"])
@pytest.mark.parametrize(
    "allocator", ["water_filling", "proportional", "power", "volume", "uniform"]
)
@pytest.mark.parametrize("refiner", ["fft", "maximin", "local_swap", "none"])
@pytest.mark.parametrize("degenerate", [False, True])
def test_pipeline(partitioner, allocator, refiner, degenerate):
    X = (
        np.zeros((24, 3))
        if degenerate
        else np.random.default_rng(23).normal(size=(24, 3))
    )
    kw = dict(
        partitioner=partitioner,
        allocator=allocator,
        refiner=refiner,
        n_blocks=4,
        r=2,
        d_intrinsic=3,
        seed=4,
    )
    a = sketch(X, 5, **kw)
    b = sketch(X, 5, **kw)
    assert np.array_equal(a.indices, b.indices)
    assert len(a.indices) == len(np.unique(a.indices)) == 5
    assert a.pool_indices.size == a.budget == a.m_per_block.sum()
    assert np.isin(a.indices, a.pool_indices).all()
    assert np.array_equal(np.unique(a.block_labels), np.arange(a.n_blocks))


@pytest.mark.parametrize("N", [1, 2, 9])
@pytest.mark.parametrize("partitioner", ["kmeans", "none", "random", "pc_tree"])
def test_select_all(N, partitioner):
    X = np.zeros((N, 2))
    res = sketch(X, N, partitioner=partitioner, certify=True)
    np.testing.assert_array_equal(res.indices, np.arange(N))
    assert res.radius == res.opt_lower == 0


@pytest.mark.parametrize("n", [True, 2.1, 0, -1, 11])
def test_invalid_n(n):
    with pytest.raises(ValueError):
        sketch(np.ones((10, 2)), n)


@pytest.mark.parametrize(
    "X",
    [
        np.ones((4, 0)),
        np.ones(4),
        np.ones((0, 2)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1j]]),
    ],
)
def test_invalid_matrix(X):
    with pytest.raises(ValueError):
        sketch(X, 1, partitioner="none")


@pytest.mark.parametrize(
    "kw",
    [
        {"r": np.nan},
        {"r": np.inf},
        {"mix": -1},
        {"mix": np.nan},
        {"selector": "random"},
        {"selector": "bad"},
        {"n_blocks": 2.1},
        {"alpha": np.nan},
        {"d_intrinsic": np.nan},
        {"refiner": "bad"},
        {"mix": 0.5, "refiner": "maximin"},
        {"mix": 0.5, "refiner": "none"},
    ],
)
def test_invalid_options(kw):
    with pytest.raises(ValueError):
        sketch(np.ones((10, 2)), 2, **kw)


@pytest.mark.parametrize("offset", [0, 1e9])
def test_geometry(offset):
    X = np.arange(12.0).reshape(6, 2) + offset
    idx, st = farthest_first(X, 6, return_state=True)
    assert len(np.unique(idx)) == 6
    assert st.radius == 0
    np.testing.assert_allclose(
        st.radii, [cdist(X, X[idx[:k]]).min(axis=1).max() for k in range(1, 7)]
    )
    for chunk in [1, 3, 4096]:
        d, j = assign_nearest(X, X[[0, 3]], chunk)
        np.testing.assert_allclose(d, cdist(X, X[[0, 3]]).min(axis=1))
    assert (
        covering_radius(np.array([[offset], [offset + 1]]), np.array([[offset]])) == 1
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bruteforce_certificate(seed):
    X = np.random.default_rng(seed).random((7, 2))
    n = 2
    optimum = min(
        cdist(X, X[list(ids)]).min(axis=1).max()
        for ids in itertools.combinations(range(7), n)
    )
    b = opt_bounds(X, n, seed=seed)
    assert b["opt_lower"] <= optimum + 1e-12 <= b["opt_upper"] + 1e-12
    res = sketch(X, n, partitioner="random", n_blocks=2, r=2, certify=True, seed=seed)
    assert res.radius / optimum <= res.ratio_upper + 1e-12
    assert res.radius <= res.theory_bound + 1e-12


def test_cache_and_claim():
    X = np.arange(3.0)[:, None]
    with pytest.raises(ValueError):
        certificate(X, [1], opt_cache=opt_bounds(100 * X, 1))
    assert certificate(X, [1], opt_cache=opt_bounds(X, 1))["ratio_upper"] >= 1
    with pytest.raises(ValueError):
        certificate(X, [0.1])
    with pytest.raises(ValueError):
        certificate(X, [0], pool_indices=[1])
    c = certificate(
        np.arange(101.0)[:, None], np.arange(50), pool_indices=np.arange(101)
    )
    assert np.isnan(c["theory_bound"]) and c["ratio_upper"] == 102


def test_unfunded_and_allocation():
    info = water_filling(
        np.arange(3.0)[:, None], np.arange(3), 3, 1, np.random.default_rng(0)
    )[2]
    assert np.isinf(info["rho_blockwise_max"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        caps = rng.integers(0, 20, size=8)
        budget = int(rng.integers(0, int(caps.sum()) + 1))
        quotas = round_to_budget(rng.random(8), budget, caps)
        assert quotas.sum() == budget and (quotas >= 0).all() and (quotas <= caps).all()
    with pytest.raises(ValueError):
        round_to_budget([np.nan, 1], 3, [5, 5])


def test_seeded_and_mix():
    X = np.zeros((20, 2))
    idx = farthest_first_seeded(X, 5, [2, 4])
    assert len(np.unique(idx)) == 5 and {2, 4} <= set(idx)
    with pytest.raises(ValueError):
        farthest_first_seeded(X, 1, [2, 4])
    for source in ["pool", "data"]:
        for mix in [0.2, 0.5, 1]:
            a = sketch(
                X, 5, partitioner="none", r=1, mix=mix, mix_source=source, seed=8
            )
            assert len(np.unique(a.indices)) == 5 and a.config["mix"] == mix


def test_pc_tree_ties():
    X = np.array([0.0] * 90 + [1.0] * 10)[:, None]
    labels = pc_tree_blocks(X, 2)
    assert list(np.bincount(labels)) == [50, 50]


def test_scsampler_adapter(monkeypatch):
    seen = []

    def fake(X, **kw):
        seen.append(kw["random_state"])
        np.random.seed(kw["random_state"])
        return np.random.choice(len(X), size=kw["n_obs"], replace=False)

    monkeypatch.setitem(sys.modules, "scsampler", types.SimpleNamespace(scsampler=fake))
    X = np.arange(30.0)[:, None]
    np.random.seed(123)
    before = np.random.get_state()
    a = select_scsampler(X, 4, np.random.default_rng(11))
    b = select_scsampler(X, 4, np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)
    assert seen[0] == seen[1]
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]

    def broken(*args, **kwargs):
        raise RuntimeError("upstream failure")

    monkeypatch.setitem(
        sys.modules, "scsampler", types.SimpleNamespace(scsampler=broken)
    )
    with pytest.raises(RuntimeError):
        select_scsampler(X, 4, np.random.default_rng(0))


def test_choose_r():
    X = np.zeros((10, 2))
    a = choose_r(X, 2, partitioner="none", r_grid=(1.0, 2.0))
    assert a["tolerance_met"] and a["r"] == 1 and len(a["trajectory"]) == 1
    for grid in [(), (2, 1), (np.nan,), (0,)]:
        with pytest.raises(ValueError):
            choose_r(X, 2, r_grid=grid)


def test_dict_semantics():
    r = SketchResult(indices=np.array([0]))
    assert not hasattr(r, "missing")
    assert copy.deepcopy(r).indices[0] == 0


def test_anndata(tmp_path, monkeypatch):
    ad = pytest.importorskip("anndata")
    from scipy.sparse import csr_matrix

    a = ad.AnnData(np.ones((20, 4)))
    a.obsm["X_pca"] = np.random.default_rng(0).random((20, 2))
    kw = dict(partitioner="none")
    sub = sketch_adata(a, 4, return_adata=True, **kw)
    assert sub.shape == (4, 4) and not sub.is_view and "pros" in sub.uns
    sub.X[0, 0] = 99
    assert a.X[0, 0] == 1
    sub.write_h5ad(tmp_path / "subset.h5ad")
    assert ad.read_h5ad(tmp_path / "subset.h5ad").n_obs == 4
    view = sketch_adata(a, 4, return_adata=True, copy=False, **kw)
    assert view.is_view
    a.write_h5ad(tmp_path / "in.h5ad")
    opened = []
    original = ad.read_h5ad

    def spy(*args, **kwargs):
        obj = original(*args, **kwargs)
        opened.append(obj)
        return obj

    monkeypatch.setattr(ad, "read_h5ad", spy)
    for returns in [False, True]:
        sketch_adata(tmp_path / "in.h5ad", 4, return_adata=returns, **kw)
        assert not opened[-1].file.is_open
    with pytest.raises(KeyError):
        sketch_adata(tmp_path / "in.h5ad", 4, use_rep="missing", **kw)
    assert not opened[-1].file.is_open
    with pytest.raises(ValueError):
        sketch_adata(tmp_path / "in.h5ad", 4, return_adata=True, copy=False, **kw)
    assert not opened[-1].file.is_open
    a.X = csr_matrix(a.X)
    with pytest.raises(ValueError):
        sketch_adata(a, 4, use_rep="X", **kw)


def test_local_search_monotonic():
    for seed in range(10):
        X = np.random.default_rng(seed).random((30, 3))
        before = farthest_first(X, 4)
        after = kcenter_local_search(X, before)
        assert covering_radius(X, X[after]) <= covering_radius(X, X[before]) + 1e-12
