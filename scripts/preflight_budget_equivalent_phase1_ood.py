"""CPU-only Phase 1 preflight including frozen arithmetic OOD bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    read_json_object,
    resolve_phase1_contract,
    validate_matrix_config,
)
from eg_sft.experiment.budget_equivalent_protocol import repository_path  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json_object(config_path)
    validate_matrix_config(config)
    ood = config.get("ood_evaluation")
    if not isinstance(ood, dict) or ood.get("required_before_unblinding") is not True:
        raise ValueError("formal OOD binding is missing")
    manifest_path = repository_path(ROOT, str(ood["manifest"]["path"]))
    if file_sha256(manifest_path) != ood["manifest"]["sha256"]:
        raise ValueError("OOD manifest hash changed")
    dataset_audits = []
    for name, binding in ood["datasets"].items():
        path = repository_path(ROOT, str(binding["path"]))
        rows = read_jsonl(path)
        ready = (
            file_sha256(path) == binding["sha256"]
            and len(rows) == int(binding["expected_record_count"])
            and all(str(row.get("dataset")) == name for row in rows)
            and all("question" not in row and "answer" not in row for row in rows)
        )
        if not ready:
            raise ValueError(f"OOD binding failed: {name}")
        dataset_audits.append(
            {"dataset": name, "status": "READY", "record_count": len(rows)}
        )
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
                "selection_sha256": contract["selection"]["file_sha256"],
            }
        )
    payload = {
        "schema_version": "budget-equivalent-phase1-ood-contract-audit-v1",
        "status": "PASS",
        "config_sha256": file_sha256(config_path),
        "ready_cell_count": len(cells),
        "ood_datasets": dataset_audits,
        "cells": cells,
        "gpu_accessed": False,
        "accuracy_withheld": True,
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
