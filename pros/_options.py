"""User-facing option metadata for the public sketch API."""

from __future__ import annotations

from typing import Literal

Partitioner = Literal["kmeans", "pc_tree", "random", "none"]
NBlocks = int | Literal["auto"]
Allocator = Literal["water_filling", "proportional", "power", "volume", "uniform"]
Selector = Literal["fft", "scsampler_maximin", "random"]
Refiner = Literal["fft", "maximin", "local_swap", "none"]
MixSource = Literal["pool", "data"]

_OPTION_DETAILS: dict[str, tuple[tuple[str, str], ...]] = {
    "partitioner": (
        ("kmeans", "default; MiniBatchKMeans partitioning"),
        ("pc_tree", "recursive principal-component partitioning"),
        ("random", "equal-size random blocks"),
        ("none", "single global block"),
    ),
    "n_blocks": (
        ("auto", "default; empirical block-count rule"),
        ("int", "explicit positive block count, clipped to the number of rows"),
    ),
    "allocator": (
        ("water_filling", "default; geometry-aware adaptive allocation"),
        ("proportional", "proportional to block size"),
        ("power", "proportional to block size raised to alpha"),
        ("volume", "geometry-based volume allocation using d_intrinsic"),
        ("uniform", "as even as block caps allow"),
    ),
    "selector": (
        ("fft", "default; farthest-first traversal"),
        ("scsampler_maximin", "upstream scSampler maximin selection"),
        ("random", "uniform sample without replacement"),
    ),
    "refiner": (
        ("fft", "default; global farthest-first refinement"),
        ("maximin", "swap improvement for pairwise separation"),
        ("local_swap", "swap-based local search for covering radius"),
        ("none", "uniformly downsample the candidate pool"),
    ),
    "mix_source": (
        ("pool", "default; draw the uniform reserve from the candidate pool"),
        ("data", "draw the uniform reserve from the full data"),
    ),
}


def _format_category(name: str) -> list[str]:
    lines = [f"{name}:"]
    for value, description in _OPTION_DETAILS[name]:
        lines.append(f"  {value:<18} {description}")
    return lines


def options(category: str | None = None) -> None:
    """Print configurable values accepted by :func:`pros.sketch`.

    Parameters
    ----------
    category : str, optional
        One option family to show, such as ``"partitioner"`` or
        ``"allocator"``. Omit it to show every configurable category.

    Examples
    --------
    >>> import pros
    >>> pros.options("partitioner")
    partitioner:
      kmeans             default; MiniBatchKMeans partitioning
      pc_tree            recursive principal-component partitioning
      random             equal-size random blocks
      none               single global block
    """
    if category is not None:
        if category not in _OPTION_DETAILS:
            valid = ", ".join(_OPTION_DETAILS)
            raise ValueError(
                f"unknown option category {category!r}; expected one of {valid}"
            )
        print("\n".join(_format_category(category)))
        return

    blocks = ["PROS sketch options"]
    for name in _OPTION_DETAILS:
        blocks.append("")
        blocks.extend(_format_category(name))
    print("\n".join(blocks))
