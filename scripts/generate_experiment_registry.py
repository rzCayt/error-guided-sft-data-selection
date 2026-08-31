#!/usr/bin/env python3
"""Generate or verify the public 24-cell registry from the frozen matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase2_clean_common24_v8_canonical.json"
SUMMARY = ROOT / "results" / "public_summary" / "main_results.json"
OUTPUT = ROOT / "results" / "public_summary" / "experiment_registry.csv"
FIELDS = (
    "cell_id",
    "method",
    "replicate_index",
    "selection_seed",
    "train_seed",
    "selection_manifest_sha256",
    "analysis_status",
)


def build_csv() -> str:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cells = config.get("job_order")
    if not isinstance(cells, list) or len(cells) != 24:
        raise RuntimeError("frozen canonical matrix must contain exactly 24 job-order cells")
    if summary["completed_study"]["cell_count"] != len(cells):
        raise RuntimeError("canonical summary cell count does not match frozen matrix")
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise RuntimeError("duplicate cell_id in frozen matrix")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for cell in sorted(cells, key=lambda item: (item["method"], item["replicate_index"], item["train_seed"])):
        writer.writerow(
            {
                "cell_id": cell["cell_id"],
                "method": cell["method"],
                "replicate_index": cell["replicate_index"],
                "selection_seed": cell["selection_seed"],
                "train_seed": cell["train_seed"],
                "selection_manifest_sha256": cell["parent_selection_manifest_sha256"],
                "analysis_status": "included_in_audited_24_cell_analysis",
            }
        )
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build_csv()
    if args.write:
        if OUTPUT.exists():
            raise FileExistsError(f"refusing to overwrite registry: {OUTPUT}")
        OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
        print("WRITTEN cells=24")
        return 0
    actual = OUTPUT.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError("public experiment registry does not match frozen matrix")
    print("PASS cells=24")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
