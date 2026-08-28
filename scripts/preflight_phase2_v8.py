"""CPU-only P0 gate for the clean 24-cell v8 long experiment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    write_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract  # noqa: E402
from eg_sft.experiment.phase2_clean_common_v8 import (  # noqa: E402
    validate_clean_common_matrix,
)
from eg_sft.experiment.phase2_v7_control import Phase2StateStore, worker_schedule  # noqa: E402
from eg_sft.experiment.phase2_v8_canonical_runtime import (  # noqa: E402
    require_canonical_role,
    validate_canonical_runtime,
)
from eg_sft.experiment.phase2_v8_contract_audit import resolved_contract_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_clean_common24_v8_canonical.json")
    )
    parser.add_argument("--canonical-runtime-files", type=Path, required=True)
    parser.add_argument("--precision-simulation", type=Path, required=True)
    parser.add_argument("--parent-evidence-index", type=Path, required=True)
    parser.add_argument("--training-input-contract-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    args = parser.parse_args()
    config_path = args.config.resolve()
    matrix = read_json(config_path)
    validate_clean_common_matrix(matrix)
    canonical = validate_canonical_runtime(
        repo_root=ROOT, manifest_path=args.canonical_runtime_files.resolve()
    )
    require_canonical_role(canonical=canonical, role="primary_matrix", actual_path=config_path)
    parent_path = (ROOT / matrix["parent_matrix"]["path"]).resolve()
    resolved = []
    for job in matrix["job_order"]:
        child = resolve_phase1_contract(repo_root=ROOT, config_path=config_path, cell_id=job["cell_id"])
        parent = resolve_phase1_contract(repo_root=ROOT, config_path=parent_path, cell_id=job["parent_cell_id"])
        diff = resolved_contract_evidence(child=child, parent=parent)
        resolved.append(
            {
                "cell_id": child["cell_id"],
                "parent_cell_id": parent["cell_id"],
                "train_seed": child["seed"],
                "selection_manifest_sha256": child["selection"]["file_sha256"],
                "selected_id_sha256": child["selection"]["selected_id_sha256"],
                "static_contract_diff_status": diff["status"],
            }
        )
    precision = read_json(args.precision_simulation.resolve())
    if precision.get("status") != "PASS" or precision.get("equivalence_status") != "EXPLORATORY_ONLY":
        raise ValueError("v8 precision simulation or equivalence downgrade changed")
    parent_evidence = read_json(args.parent_evidence_index.resolve())
    if parent_evidence.get("status") != "PASS" or int(parent_evidence.get("parent_cell_count", -1)) != 16:
        raise ValueError("v8 parent evidence index is incomplete")
    input_contract_status = "PENDING_MODEL_SNAPSHOT_ON_DATA_DISK"
    input_contract_sha = None
    if args.training_input_contract_root is not None:
        complete = args.training_input_contract_root.resolve() / "MATERIALIZATION_COMPLETE.json"
        require_canonical_role(
            canonical=canonical, role="materialized_contracts", actual_path=complete
        )
        materialized = read_json(complete)
        if (
            materialized.get("status") != "PASS"
            or int(materialized.get("cell_count", -1)) != 24
            or materialized.get("config_sha256") != file_sha256(config_path)
        ):
            raise ValueError("v8 tokenized training contracts are incomplete")
        input_contract_status = "PASS"
        input_contract_sha = file_sha256(complete)
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < args.min_free_gib:
        raise ValueError("v8 local disk preflight failed")
    output_root = args.output_root.resolve()
    store = Phase2StateStore(root=output_root / "control_template", matrix_path=config_path)
    state_report = store.initialize()
    commands = [
        {
            "worker_id": worker,
            "position": position,
            "cell_id": cell,
            "contract_only_command": (
                "python scripts/run_identifiable_budget_v4_cell.py --config "
                "configs/phase2_clean_common24_v8_canonical.json "
                f"--cell-id {cell} --contract-only"
            ),
        }
        for worker in ("gpu0", "gpu1")
        for position, cell in enumerate(worker_schedule(matrix, worker), start=1)
    ]
    report = {
        "schema_version": "phase2-v8-cpu-preflight-v1",
        "status": "PASS" if input_contract_status == "PASS" else "PASS_PENDING_REMOTE_STATIC_MATERIALIZATION",
        "matrix_sha256": file_sha256(config_path),
        "canonical_runtime_sha256": canonical["manifest_sha256"],
        "resolved_cell_count": len(resolved),
        "resolved_cells": resolved,
        "worker_cell_counts": {worker: len(worker_schedule(matrix, worker)) for worker in ("gpu0", "gpu1")},
        "training_input_contract_status": input_contract_status,
        "training_input_materialization_sha256": input_contract_sha,
        "precision_simulation_sha256": file_sha256(args.precision_simulation.resolve()),
        "equivalence_status": "EXPLORATORY_ONLY",
        "parent_evidence_index_sha256": file_sha256(args.parent_evidence_index.resolve()),
        "state_initialization": state_report,
        "command_count": len(commands),
        "free_disk_gib": free_gib,
        "gpu_accessed": False,
        "accuracy_withheld": True,
    }
    write_exclusive_or_verify(output_root / "preflight_report.json", canonical_json_bytes(report))
    write_exclusive_or_verify(
        output_root / "commands.jsonl",
        b"".join(canonical_json_bytes(row) for row in commands),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
