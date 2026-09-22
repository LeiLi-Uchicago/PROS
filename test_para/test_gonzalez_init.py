"""Experiment 1: Gonzalez initialization sensitivity."""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pros import opt_bounds  # noqa: E402

from datasets import DATASETS, ensure_datasets, load_dataset  # noqa: E402


N = 100
SEEDS = tuple(range(20))
OUT_DIR = Path(__file__).resolve().parent / "results"


RUN_COLUMNS = [
    "dataset",
    "n",
    "seed",
    "delta",
    "opt_lower",
    "opt_upper",
    "fft_radius_n",
    "runtime",
]

SUMMARY_COLUMNS = [
    "dataset",
    "n",
    "num_seeds",
    "delta_mean",
    "delta_std",
    "delta_cv",
    "delta_min",
    "delta_max",
    "delta_range",
    "delta_range_ratio",
    "opt_lower_mean",
    "opt_lower_std",
    "opt_lower_cv",
    "opt_lower_min",
    "opt_lower_max",
    "opt_lower_range",
    "opt_lower_range_ratio",
]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)))


def _stats(prefix: str, values: list[float]) -> dict[str, float]:
    mean = _mean(values)
    std = _std(values)
    min_value = float(min(values))
    max_value = float(max(values))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": std / mean if mean != 0.0 else float("nan"),
        f"{prefix}_min": min_value,
        f"{prefix}_max": max_value,
        f"{prefix}_range": max_value - min_value,
        f"{prefix}_range_ratio": max_value / min_value if min_value != 0.0 else float("nan"),
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(columns: list[str], rows: list[dict]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def run() -> tuple[list[dict], list[dict]]:
    ensure_datasets()
    run_rows: list[dict] = []
    summary_rows: list[dict] = []

    for spec in DATASETS:
        X, _ = load_dataset(spec)
        for seed in SEEDS:
            start = time.perf_counter()
            bounds = opt_bounds(X, N, seed=seed)
            runtime = time.perf_counter() - start
            run_rows.append(
                {
                    "dataset": spec.name,
                    "n": N,
                    "seed": seed,
                    "delta": float(bounds["delta"]),
                    "opt_lower": float(bounds["opt_lower"]),
                    "opt_upper": float(bounds["opt_upper"]),
                    "fft_radius_n": float(bounds["fft_radius_n"]),
                    "runtime": runtime,
                }
            )

        dataset_rows = [row for row in run_rows if row["dataset"] == spec.name]
        summary = {
            "dataset": spec.name,
            "n": N,
            "num_seeds": len(SEEDS),
        }
        summary.update(_stats("delta", [float(row["delta"]) for row in dataset_rows]))
        summary.update(
            _stats("opt_lower", [float(row["opt_lower"]) for row in dataset_rows])
        )
        summary_rows.append(summary)

    return run_rows, summary_rows


def write_outputs(run_rows: list[dict], summary_rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / "gonzalez_init_runs.csv", run_rows, RUN_COLUMNS)
    _write_csv(OUT_DIR / "gonzalez_init_summary.csv", summary_rows, SUMMARY_COLUMNS)

    md = [
        "# Gonzalez Initialization Sensitivity",
        "",
        f"Fixed `n={N}` and seeds `{SEEDS[0]}..{SEEDS[-1]}`.",
        "",
        _markdown_table(SUMMARY_COLUMNS, summary_rows),
        "",
    ]
    (OUT_DIR / "gonzalez_init_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    run_rows, summary_rows = run()
    write_outputs(run_rows, summary_rows)
    print(f"Wrote {len(run_rows)} run rows and {len(summary_rows)} summary rows.")
    print(f"Results: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

