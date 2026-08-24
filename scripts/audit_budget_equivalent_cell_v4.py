"""Write one immutable, blind-safe v4 CPU audit for a Phase 1 cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_audit_v4 import (  # noqa: E402
    audit_phase1_run_v4,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_phase1_run_v4(
        repo_root=ROOT,
        config_path=args.config.resolve(),
        cell_id=args.cell_id,
        run_dir=args.run_dir.resolve(),
    )
    output = args.run_dir.resolve() / "audit" / "formal_cell_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    sidecar = output.with_suffix(".sha256")
    sidecar.write_text(f"{file_sha256(output)}  {output.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "status": "PASS",
                "cell_id": args.cell_id,
                "audit_sha256": file_sha256(output),
                "parser_rows_recomputed": report["evaluation"][
                    "parser_rows_recomputed_from_raw_text"
                ],
                "accuracy_withheld": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
