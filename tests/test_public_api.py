from __future__ import annotations

import inspect

import pytest

import pros


def test_public_api_is_small_and_documented() -> None:
    assert set(pros.__all__) == {
        "sketch",
        "sketch_adata",
        "SketchResult",
        "certificate",
        "opt_bounds",
        "choose_r",
        "estimate_opt_scale",
        "options",
        "__version__",
    }
    assert not hasattr(pros, "water_filling")
    assert not hasattr(pros, "make_partition")


def test_sketch_signature_uses_categorical_type_aliases() -> None:
    sig = inspect.signature(pros.sketch)

    assert sig.parameters["partitioner"].annotation == "Partitioner"
    assert sig.parameters["n_blocks"].annotation == "NBlocks"
    assert sig.parameters["allocator"].annotation == "Allocator"
    assert sig.parameters["selector"].annotation == "Selector"
    assert sig.parameters["refiner"].annotation == "Refiner"
    assert sig.parameters["mix_source"].annotation == "MixSource"


def test_options_prints_all_categories(capsys: pytest.CaptureFixture[str]) -> None:
    pros.options()

    out = capsys.readouterr().out
    assert "PROS sketch options" in out
    assert "partitioner:" in out
    assert "allocator:" in out
    assert "selector:" in out
    assert "refiner:" in out
    assert "mix_source:" in out
    assert "water_filling" in out


def test_options_prints_one_category(capsys: pytest.CaptureFixture[str]) -> None:
    pros.options("partitioner")

    out = capsys.readouterr().out
    assert out.startswith("partitioner:")
    assert "kmeans" in out
    assert "allocator:" not in out


def test_options_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="unknown option category"):
        pros.options("bad")
