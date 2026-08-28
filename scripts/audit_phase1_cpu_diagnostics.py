#!/usr/bin/env python3
"""Create a non-overwriting CPU-only Phase 1A diagnostic artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from eg_sft.experiment.cpu_identifiability_audit import (
    run_cpu_identifiability_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Evidence .tar.gz, evidence directory, or extracted cell directory; repeatable.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--near-duplicate-clusters", type=Path)
    parser.add_argument("--expected-cells", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=20260827)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--allow-nonformal-dataset-counts",
        action="store_true",
        help="Testing/debug only: do not require 1319/300/2067/155 rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_cpu_identifiability_audit(
        inputs=args.input,
        output_root=args.output_root,
        selection_root=args.selection_root,
        near_duplicate_clusters=args.near_duplicate_clusters,
        expected_cells=args.expected_cells,
        expected_dataset_counts=not args.allow_nonformal_dataset_counts,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        run_id=args.run_id,
    )
    print(output)


if __name__ == "__main__":
    main()

