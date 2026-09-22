from __future__ import annotations

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
        "__version__",
    }
    assert not hasattr(pros, "water_filling")
    assert not hasattr(pros, "make_partition")
