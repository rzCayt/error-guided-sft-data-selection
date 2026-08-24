"""Validate every frozen Phase 1 cell without touching CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    read_json_object,
    resolve_phase1_contract,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json_object(config_path)
    cells = []
    for job in config["job_order"]:
        contract = resolve_phase1_contract(
            repo_root=ROOT,
            config_path=config_path,
            cell_id=str(job["cell_id"]),
        )
        cells.append(
            {
                "cell_id": contract["cell_id"],
                "status": "READY",
                "config_sha256": contract["config_sha256"],
                "selection_sha256": contract["selection"]["file_sha256"],
                "selected_id_sha256": contract["selection"]["selected_id_sha256"],
            }
        )
    payload = {
        "schema_version": "budget-equivalent-phase1-contract-audit-v1",
        "status": "PASS" if len(cells) == 16 else "FAIL",
        "config_sha256": file_sha256(config_path),
        "ready_cell_count": len(cells),
        "cells": cells,
        "accuracy_withheld": True,
        "gpu_accessed": False,
    }
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
