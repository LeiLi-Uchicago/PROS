"""Run all PROS parameter experiments."""

from __future__ import annotations

from datasets import ensure_datasets
from test_gonzalez_init import main as run_gonzalez_init
from compare_partition_allocation import main as run_partition_allocation


def main() -> int:
    ensure_datasets()
    print("Running experiment 1: Gonzalez initialization sensitivity")
    status = run_gonzalez_init()
    if status != 0:
        return status

    print("Running experiment 2: partition/allocation comparison")
    status = run_partition_allocation()
    if status != 0:
        return status

    print("Done. Results are in test_para/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

