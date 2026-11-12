import sys
import types

import numpy as np
import pytest

import pros
from pros.select import refine, select_geosketch


def assert_rng_restored(before):
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize(
    ("selector", "refiner"),
    [("geosketch", "fft"), ("fft", "geosketch"), ("geosketch", "geosketch")],
)
def test_native_stages(monkeypatch, selector, refiner):
    upstream = pytest.importorskip("geosketch")
    gs = upstream.gs
    calls = []

    def record(data, count, *, replace, seed):
        assert replace is False
        rows = gs(data, count, replace=replace, seed=seed)
        calls.append((data.copy(), rows, seed))
        return rows

    monkeypatch.setattr(upstream, "gs", record)
    x = np.random.default_rng(33).normal(size=(120, 5))
    opts = dict(
        allocator="power",
        partitioner="random",
        n_blocks=3,
        r=2,
        selector=selector,
        refiner=refiner,
        seed=17,
        certify=True,
    )
    before = np.random.get_state()
    result = pros.sketch(x, 12, **opts)
    assert_rng_restored(before)
    assert result.config["selector"] == selector and result.config["refiner"] == refiner
    assert result.indices.dtype == np.int64
    assert len(np.unique(result.indices)) == 12
    assert np.isin(result.indices, result.pool_indices).all()
    assert len(calls) == (3 if selector == "geosketch" else 0) + (
        refiner == "geosketch"
    )
    if selector == "geosketch":
        selected = []
        for i, (data, positions, _) in enumerate(calls[:3]):
            members = np.flatnonzero(result.block_labels == i)
            np.testing.assert_array_equal(data, x[members])
            selected.extend(members[positions])
        np.testing.assert_array_equal(result.pool_indices, np.sort(selected))
        assert len({call[2] for call in calls[:3]}) == 3
    if refiner == "geosketch":
        pool, positions, seed = calls[-1]
        assert seed == opts["seed"]
        np.testing.assert_array_equal(pool, x[result.pool_indices])
        np.testing.assert_array_equal(
            result.indices, np.sort(result.pool_indices[positions])
        )
        assert np.isnan(result.theory_bound)
    again = pros.sketch(x, 12, **opts)
    np.testing.assert_array_equal(result.pool_indices, again.pool_indices)
    np.testing.assert_array_equal(result.indices, again.indices)
    assert_rng_restored(before)


@pytest.mark.parametrize("role", ["selector", "refiner"])
def test_geosketch_optional_and_discoverable(role, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "geosketch", None)
    x = np.arange(120.0).reshape(40, 3)
    pros.options(role)
    assert "geosketch" in capsys.readouterr().out
    assert pros.sketch(x, 4).indices.size == 4
    with pytest.raises(ImportError, match=r"pros-sketch\[geosketch\]"):
        pros.sketch(x, 4, allocator="power", **{role: "geosketch"})


@pytest.mark.parametrize("returned", [[0, 0], [0], [0, 10], [0.5, 1.5], "fail"])
def test_upstream_failure_and_rng_restoration(monkeypatch, returned):
    def broken(data, count, *, replace, seed):
        np.random.seed(seed)
        if returned == "fail":
            raise RuntimeError("upstream failure")
        return returned

    monkeypatch.setitem(sys.modules, "geosketch", types.SimpleNamespace(gs=broken))
    before = np.random.get_state()
    with pytest.raises(
        RuntimeError if returned == "fail" else ValueError,
        match="upstream failure|geosketch",
    ):
        select_geosketch(np.arange(10)[:, None], 2, np.random.default_rng(31))
    assert_rng_restored(before)


def test_geosketch_constraints():
    x = np.arange(60.0).reshape(20, 3)
    with pytest.raises(ValueError, match="water_filling"):
        pros.sketch(x, 4, selector="geosketch")
    with pytest.raises(ValueError, match="mix"):
        pros.sketch(x, 4, refiner="geosketch", mix=0.1)
    with pytest.raises(ValueError, match="mix"):
        refine(x, 4, "geosketch", np.random.default_rng(0), mix=0.1)


@pytest.mark.parametrize("n", [1, 5, 12])
def test_constant_coordinates_and_full_quota(n):
    pytest.importorskip("geosketch")
    x = np.full((12, 2), 25.0)
    opts = dict(
        allocator="power",
        partitioner="random",
        n_blocks=2,
        selector="geosketch",
        refiner="geosketch",
        r=2,
        seed=2**80,
    )
    with np.errstate(all="raise"):
        result = pros.sketch(x, n, **opts)
        again = pros.sketch(x, n, **opts)
    assert len(np.unique(result.indices)) == n
    np.testing.assert_array_equal(result.indices, again.indices)
    if n == len(x):
        np.testing.assert_array_equal(result.indices, np.arange(n))
