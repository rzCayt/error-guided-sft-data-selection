"""CPU-resolvable contract for the pre-Phase-1 cloud qualification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eg_sft.experiment.budget_equivalent_matrix import (
    read_json_object,
    resolve_frozen_file,
    validate_matrix_config,
)
from eg_sft.experiment.budget_equivalent_ood_runtime import (
    OOD_DATASETS,
    resolve_ood_contract,
)
from eg_sft.experiment.budget_equivalent_protocol import repository_path
from eg_sft.training.b500 import file_sha256, read_jsonl


def resolve_qualification_contract(
    *, repo_root: Path, qualification_config_path: Path
) -> dict[str, Any]:
    """Validate every frozen input without importing or touching CUDA."""

    qualification_config_path = qualification_config_path.resolve()
    qualification = read_json_object(qualification_config_path)
    if qualification.get("protocol_version") != "budget-equivalent-runtime-qualification-v2":
        raise ValueError("unexpected qualification protocol version")
    matrix_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=qualification["matrix_config"],
        label="qualification matrix config",
    )
    matrix = read_json_object(matrix_path)
    validate_matrix_config(matrix)
    if qualification["qualification_data"].get("formal_phase1_selection_consumed") is not False:
        raise ValueError("qualification must not consume a formal Phase 1 selection")

    protocol_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=matrix["protocol_config"],
        label="protocol config",
    )
    protocol = read_json_object(protocol_path)
    data_dir = repository_path(repo_root, str(matrix["data_manifest"]["directory"]))
    for filename, expected in matrix["data_manifest"]["required_files"].items():
        path = data_dir / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"qualification data file changed: {filename}")
    gsm_records = read_jsonl(data_dir / "gsm8k_records.jsonl")

    data_spec = qualification["qualification_data"]
    overfit = sorted(
        (row for row in gsm_records if row["protocol_split"] == data_spec["overfit_split"]),
        key=lambda row: (row["source_index"], row["record_id"]),
    )[: int(data_spec["overfit_example_count"])]
    canary = sorted(
        (row for row in gsm_records if row["protocol_split"] == data_spec["canary_split"]),
        key=lambda row: (row["source_index"], row["record_id"]),
    )[: int(data_spec["canary_example_count"])]
    if len(overfit) != int(data_spec["overfit_example_count"]):
        raise ValueError("qualification overfit set is incomplete")
    if len(canary) != int(data_spec["canary_example_count"]):
        raise ValueError("qualification canary set is incomplete")
    if set(row["record_id"] for row in overfit) & set(row["record_id"] for row in canary):
        raise ValueError("qualification overfit and canary sets overlap")

    gates = qualification["single_gpu_gates"]
    required_gates = {
        "bf16_required": True,
        "canary_output_count": 128,
        "parser_recompute_must_match": True,
        "checkpoint_resume_required": True,
        "optimizer_scheduler_rng_restore_required": True,
        "output_directory_must_be_non_overwriting": True,
        "ood_contract_preflight_required": True,
    }
    for field, expected in required_gates.items():
        if gates.get(field) != expected:
            raise ValueError(f"qualification gate changed: {field}")
    ood_contracts = {
        dataset: resolve_ood_contract(
            repo_root=repo_root,
            matrix_config_path=matrix_path,
            dataset=dataset,
        )
        for dataset in OOD_DATASETS
    }
    return {
        "qualification": qualification,
        "qualification_config_path": qualification_config_path,
        "qualification_config_sha256": file_sha256(qualification_config_path),
        "matrix": matrix,
        "matrix_path": matrix_path,
        "matrix_sha256": file_sha256(matrix_path),
        "protocol": protocol,
        "protocol_path": protocol_path,
        "data_dir": data_dir,
        "overfit_records": overfit,
        "canary_records": canary,
        "ood_contracts": ood_contracts,
    }


def qualification_preflight_summary(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "READY",
        "stage": "budget_equivalent_cloud_qualification_contract",
        "qualification_config_sha256": contract["qualification_config_sha256"],
        "matrix_config_sha256": contract["matrix_sha256"],
        "overfit_example_count": len(contract["overfit_records"]),
        "canary_example_count": len(contract["canary_records"]),
        "ood_record_counts": {
            name: len(value["records"])
            for name, value in contract["ood_contracts"].items()
        },
        "formal_phase1_selection_consumed": False,
        "formal_phase1_training_started": False,
        "gpu_accessed": False,
        "claim_boundary": contract["qualification"]["claim_boundary"],
    }
