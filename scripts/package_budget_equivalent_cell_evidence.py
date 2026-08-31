"""Create a compact per-cell evidence archive after formal and OOD audits pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.experiment.cell_evidence_package import package_cell_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-log", type=Path, action="append", default=[])
    args = parser.parse_args()
    manifest, archive_sha256 = package_cell_evidence(
        run_dir=args.run_dir,
        output=args.output,
        extra_logs=args.extra_log,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "package_budget_equivalent_cell_evidence",
                "cell_id": manifest["cell_id"],
                "file_count": manifest["file_count"],
                "archive_sha256": archive_sha256,
                "accuracy_withheld": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
