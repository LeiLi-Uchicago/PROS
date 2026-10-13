import sys
import types

import numpy as np
import pytest

from pros import sketch
from pros.select import refine


@pytest.mark.parametrize("selector", ["fft", "scsampler_maximin"])
@pytest.mark.parametrize("refiner", ["fft", "scsampler_maximin"])
def test_scsampler_stages_and_index_mapping(monkeypatch, selector, refiner):
    calls = []

    def fake(X, **kw):
        calls.append((X.copy(), kw.copy()))
        np.random.seed(kw["random_state"])
        return np.random.choice(len(X), kw["n_obs"], replace=False)

    monkeypatch.setitem(sys.modules, "scsampler", types.SimpleNamespace(scsampler=fake))
    X = np.random.default_rng(0).normal(size=(60, 3))
    before = np.random.get_state()
    opts = dict(
        partitioner="random",
        n_blocks=3,
        allocator="proportional",
        r=2,
        selector=selector,
        refiner=refiner,
        seed=7,
        certify=True,
    )
    a = sketch(X, 6, **opts)
    first_calls = list(calls)
    calls.clear()
    b = sketch(X, 6, **opts)
    expected_calls = (3 if selector == "scsampler_maximin" else 0) + (
        1 if refiner == "scsampler_maximin" else 0
    )
    assert len(first_calls) == len(calls) == expected_calls
    for (data, kw), (again, again_kw) in zip(first_calls, calls, strict=True):
        np.testing.assert_array_equal(data, again)
        assert kw == again_kw
        assert kw["random_split"] == 1 and kw["copy"] is False
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.pool_indices, b.pool_indices)
    assert len(np.unique(a.indices)) == 6
    assert np.isin(a.indices, a.pool_indices).all()
    if refiner == "scsampler_maximin":
        np.testing.assert_array_equal(first_calls[-1][0], X[a.pool_indices])
        assert first_calls[-1][1]["n_obs"] == 6
        assert np.isnan(a.theory_bound)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_scsampler_refiner_propagates_upstream_failure(monkeypatch):
    def broken(*args, **kw):
        raise RuntimeError("upstream failure")

    monkeypatch.setitem(
        sys.modules, "scsampler", types.SimpleNamespace(scsampler=broken)
    )
    with pytest.raises(RuntimeError, match="upstream failure"):
        sketch(np.arange(30.0)[:, None], 4, refiner="scsampler_maximin")


def test_maximin_does_not_call_scsampler(monkeypatch):
    # A missing upstream dependency must not affect the in-package heuristic.
    monkeypatch.setitem(sys.modules, "scsampler", None)
    a = sketch(np.arange(30.0)[:, None], 4, refiner="maximin")
    assert len(a.indices) == 4
    with pytest.raises(ImportError, match="scsampler"):
        sketch(np.arange(30.0)[:, None], 4, refiner="scsampler_maximin")


@pytest.mark.parametrize("mix", [0.1, 1.0])
def test_scsampler_refiner_rejects_mix(mix):
    X = np.arange(30.0)[:, None]
    with pytest.raises(ValueError, match="mix"):
        sketch(X, 4, refiner="scsampler_maximin", mix=mix)
    with pytest.raises(ValueError, match="mix"):
        refine(X, 4, "scsampler_maximin", np.random.default_rng(0), mix=mix)
