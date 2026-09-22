"""Experiment 2: partitioner/allocation comparison."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pros import covering_radius, sketch  # noqa: E402

from datasets import DATASETS, ensure_datasets, load_dataset  # noqa: E402


N = 100
R = 5.0
N_BLOCKS = 32
SEEDS = tuple(range(20))
OUT_DIR = Path(__file__).resolve().parent / "results"

CONFIGS = (
    {
        "config_name": "random + proportional",
        "partitioner": "random",
        "allocator": "proportional",
    },
    {
        "config_name": "kmeans + proportional",
        "partitioner": "kmeans",
        "allocator": "proportional",
    },
    {
        "config_name": "kmeans + water_filling",
        "partitioner": "kmeans",
        "allocator": "water_filling",
    },
)

RUN_COLUMNS = [
    "dataset",
    "config_name",
    "partitioner",
    "allocator",
    "seed",
    "n",
    "r",
    "n_blocks",
    "pool_size",
    "partition_time",
    "stage1_time",
    "stage2_time",
    "total_runtime",
    "rho",
    "final_radius",
]

SUMMARY_COLUMNS = [
    "dataset",
    "config_name",
    "num_seeds",
    "partition_time_mean",
    "partition_time_std",
    "partition_time_cv",
    "stage1_time_mean",
    "stage1_time_std",
    "stage1_time_cv",
    "stage2_time_mean",
    "stage2_time_std",
    "stage2_time_cv",
    "total_runtime_mean",
    "total_runtime_std",
    "total_runtime_cv",
    "rho_mean",
    "rho_std",
    "rho_cv",
    "final_radius_mean",
    "final_radius_std",
    "final_radius_cv",
    "pool_size_mean",
]


def sklearn_available() -> bool:
    return importlib.util.find_spec("sklearn") is not None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)))


def _mean_std_cv(prefix: str, values: list[float]) -> dict[str, float]:
    mean = _mean(values)
    std = _std(values)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": std / mean if mean != 0.0 else float("nan"),
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


def write_missing_dependency_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    message = "\n".join(
        [
            "# Partition and Allocation Comparison",
            "",
            "This experiment was not run because `scikit-learn` is not installed.",
            "",
            "Install dependencies with:",
            "",
            '```bash',
            'python3 -m pip install -e ".[dev]"',
            '```',
            "",
            "The k-means configurations require `sklearn.cluster.MiniBatchKMeans`, so",
            "the script does not write partial or misleading comparison results.",
            "",
        ]
    )
    (OUT_DIR / "partition_allocation_summary.md").write_text(message, encoding="utf-8")


def run() -> tuple[list[dict], list[dict]]:
    ensure_datasets()
    run_rows: list[dict] = []
    summary_rows: list[dict] = []

    for spec in DATASETS:
        X, _ = load_dataset(spec)
        for config in CONFIGS:
            for seed in SEEDS:
                res = sketch(
                    X,
                    n=N,
                    r=R,
                    n_blocks=N_BLOCKS,
                    partitioner=config["partitioner"],
                    allocator=config["allocator"],
                    selector="fft",
                    refiner="fft",
                    seed=seed,
                    certify=False,
                )
                rho = covering_radius(X, X[res.pool_indices])
                final_radius = covering_radius(X, X[res.indices])
                run_rows.append(
                    {
                        "dataset": spec.name,
                        "config_name": config["config_name"],
                        "partitioner": config["partitioner"],
                        "allocator": config["allocator"],
                        "seed": seed,
                        "n": N,
                        "r": R,
                        "n_blocks": N_BLOCKS,
                        "pool_size": int(res.pool_indices.size),
                        "partition_time": float(res.timings["partition"]),
                        "stage1_time": float(res.timings["stage1"]),
                        "stage2_time": float(res.timings["stage2"]),
                        "total_runtime": float(res.timings["total"]),
                        "rho": float(rho),
                        "final_radius": float(final_radius),
                    }
                )

            group = [
                row
                for row in run_rows
                if row["dataset"] == spec.name
                and row["config_name"] == config["config_name"]
            ]
            summary = {
                "dataset": spec.name,
                "config_name": config["config_name"],
                "num_seeds": len(group),
            }
            for metric in (
                "partition_time",
                "stage1_time",
                "stage2_time",
                "total_runtime",
                "rho",
                "final_radius",
            ):
                summary.update(_mean_std_cv(metric, [float(row[metric]) for row in group]))
            summary["pool_size_mean"] = _mean([float(row["pool_size"]) for row in group])
            summary_rows.append(summary)

    return run_rows, summary_rows


def write_outputs(run_rows: list[dict], summary_rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / "partition_allocation_runs.csv", run_rows, RUN_COLUMNS)
    _write_csv(
        OUT_DIR / "partition_allocation_summary.csv",
        summary_rows,
        SUMMARY_COLUMNS,
    )

    explanation_rows = [
        {
            "config": "random + proportional",
            "interpretation": "baseline: random blocks and block-size allocation",
        },
        {
            "config": "kmeans + proportional",
            "interpretation": "isolates spatial partition quality while keeping proportional allocation",
        },
        {
            "config": "kmeans + water_filling",
            "interpretation": "adds water-filling allocation on the same k-means partition family",
        },
    ]
    md = [
        "# Partition and Allocation Comparison",
        "",
        f"Fixed `n={N}`, `r={R}`, `n_blocks={N_BLOCKS}`, and seeds `{SEEDS[0]}..{SEEDS[-1]}`.",
        "",
        "## Configuration Interpretation",
        "",
        _markdown_table(["config", "interpretation"], explanation_rows),
        "",
        "## Summary",
        "",
        _markdown_table(SUMMARY_COLUMNS, summary_rows),
        "",
    ]
    (OUT_DIR / "partition_allocation_summary.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )


def main() -> int:
    if not sklearn_available():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_missing_dependency_outputs()
        print("scikit-learn is not installed; partition/allocation comparison was not run.")
        print('Install dependencies with: python3 -m pip install -e ".[dev]"')
        print(f"Wrote missing-dependency note: {OUT_DIR / 'partition_allocation_summary.md'}")
        return 0

    run_rows, summary_rows = run()
    write_outputs(run_rows, summary_rows)
    print(f"Wrote {len(run_rows)} run rows and {len(summary_rows)} summary rows.")
    print(f"Results: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

