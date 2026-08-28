"""Frozen extension contract for the identifiable-budget follow-up study.

The v4 study deliberately reuses every immutable v3 data, selection, model,
prompt, parser and LoRA binding.  Its small extension file is therefore a
study schedule, not a second copy of the v3 protocol.  Resolving a cell first
verifies the parent matrix SHA-256 and then derives a complete runtime
contract for exactly one new cell.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from eg_sft.experiment.budget_equivalent_matrix import (
    _validate_selection,
    read_json_object,
    resolve_frozen_file,
    validate_matrix_config,
)
from eg_sft.experiment.budget_equivalent_protocol import repository_path
from eg_sft.training.b500 import file_sha256


MATRIX_VERSION = "identifiable-budget-v4-extension-v1"
ALLOWED_STUDIES = {"dose_only", "common_seed29"}


def validate_identifiable_matrix(payload: dict[str, Any]) -> None:
    if payload.get("matrix_version") != MATRIX_VERSION:
        raise ValueError("unexpected identifiable-budget matrix version")
    jobs = payload.get("job_order")
    if not isinstance(jobs, list) or len(jobs) != 12:
        raise ValueError("identifiable-budget v4 must contain exactly 12 jobs")
    cell_ids = [str(row.get("cell_id", "")) for row in jobs]
    if not all(cell_ids) or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("v4 cell IDs must be present and unique")
    parent_ids = [str(row.get("parent_cell_id", "")) for row in jobs]
    if not all(parent_ids):
        raise ValueError("every v4 job must bind one parent v3 cell")

    dose = [row for row in jobs if row.get("study") == "dose_only"]
    common = [row for row in jobs if row.get("study") == "common_seed29"]
    if len(dose) != 4 or len(common) != 8:
        raise ValueError("v4 requires four dose-only and eight common-seed29 jobs")
    if {str(row.get("study")) for row in jobs} != ALLOWED_STUDIES:
        raise ValueError("unknown v4 study arm")
    if any(
        row.get("method") != "random_free_mix"
        or int(row.get("train_seed", -1)) != 17
        or int(row.get("supervision_token_cap", -1)) != 63680
        or row.get("token_cap_policy") != "hash_uniform_v1"
        for row in dose
    ):
        raise ValueError("dose-only jobs changed")
    if {
        (int(row.get("replicate_index", -1)), str(row.get("method")))
        for row in common
    } != {
        (replicate, method)
        for replicate in range(1, 5)
        for method in ("random_common_mix", "rds_error_common_mix")
    } or any(int(row.get("train_seed", -1)) != 29 for row in common):
        raise ValueError("common-seed29 jobs changed")

    policy = payload.get("execution_policy", {})
    required = {
        "accuracy_blind_until_all_audits": True,
        "one_cell_per_gpu": True,
        "ddp_forbidden": True,
        "formal_and_ood_audit_required": True,
        "automatic_unblinding": False,
    }
    for field, expected in required.items():
        if policy.get(field) != expected:
            raise ValueError(f"v4 execution policy changed: {field}")
    if int(policy.get("required_audited_cells", -1)) != 12:
        raise ValueError("v4 audited-cell gate changed")
    if not str(payload.get("output_root", "")).startswith(".aris/"):
        raise ValueError("v4 runtime output must stay below .aris")


def _parent_contract(
    *, repo_root: Path, config_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    extension = read_json_object(config_path)
    validate_identifiable_matrix(extension)
    parent_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=extension["parent_matrix"],
        label="identifiable-budget parent matrix",
    )
    parent = read_json_object(parent_path)
    validate_matrix_config(parent)
    return extension, parent_path, parent


def materialize_runtime_matrix(
    *, repo_root: Path, config_path: Path
) -> dict[str, Any]:
    """Return a complete matrix-shaped view for shared OOD/evaluation code."""

    extension, _, parent = _parent_contract(
        repo_root=repo_root, config_path=config_path.resolve()
    )
    runtime = deepcopy(parent)
    runtime["phase1_protocol_version"] = MATRIX_VERSION
    runtime["output_root"] = str(extension["output_root"])
    runtime["job_order"] = deepcopy(extension["job_order"])
    runtime["execution_policy"] = {
        **runtime.get("execution_policy", {}),
        "required_audited_cells_before_unblinding": 12,
        "accuracy_blind_until_all_audits": True,
        "automatic_next_cell": False,
        "one_cell_per_invocation": True,
    }
    runtime["identifiable_budget_extension"] = {
        "matrix_version": MATRIX_VERSION,
        "extension_config_sha256": file_sha256(config_path.resolve()),
    }
    return runtime


def resolve_identifiable_contract(
    *, repo_root: Path, config_path: Path, cell_id: str
) -> dict[str, Any]:
    config_path = config_path.resolve()
    extension, _, parent = _parent_contract(
        repo_root=repo_root, config_path=config_path
    )
    matching = [row for row in extension["job_order"] if row["cell_id"] == cell_id]
    if len(matching) != 1:
        raise ValueError("requested v4 cell is absent or duplicated")
    job = matching[0]
    parent_matching = [
        row for row in parent["job_order"] if row["cell_id"] == job["parent_cell_id"]
    ]
    if len(parent_matching) != 1:
        raise ValueError("v4 parent cell binding is absent or duplicated")
    parent_job = parent_matching[0]
    if (
        str(parent_job["method"]) != str(job["method"])
        or int(parent_job["replicate_index"]) != int(job["replicate_index"])
    ):
        raise ValueError("v4 job method/replicate differs from its parent selection")

    runtime = materialize_runtime_matrix(repo_root=repo_root, config_path=config_path)
    runtime_training = runtime["training"]
    runtime_training["supervision_token_cap"] = job.get("supervision_token_cap")
    runtime_training["token_cap_policy"] = job.get("token_cap_policy")
    runtime_training["tokens_per_optimizer_step"] = (
        995 if job["study"] == "dose_only" else None
    )

    protocol_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=parent["protocol_config"],
        label="protocol config",
    )
    recipe_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=parent["base_recipe_config"],
        label="base recipe",
    )
    gate_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=parent["information_gates"],
        label="information gates",
    )
    gates = read_json_object(gate_path)
    if gates.get("targeted_policy_gate_passed") is not True:
        raise ValueError("targeted-policy gate does not permit the follow-up")
    if gates.get("formal_near_duplicate_control") is not True:
        raise ValueError("near-duplicate control did not pass")

    data_dir = repository_path(repo_root, str(parent["data_manifest"]["directory"]))
    for filename, expected_hash in parent["data_manifest"]["required_files"].items():
        path = data_dir / filename
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"frozen data file changed: {filename}")
    selection = _validate_selection(repo_root=repo_root, job=parent_job)
    return {
        "config": runtime,
        "config_path": config_path,
        "config_sha256": file_sha256(config_path),
        "protocol_path": protocol_path,
        "protocol": read_json_object(protocol_path),
        "base_recipe_path": recipe_path,
        "base_recipe": read_json_object(recipe_path),
        "gate_path": gate_path,
        "data_dir": data_dir,
        "selection": selection,
        "output_root": repository_path(repo_root, str(extension["output_root"])),
        "method": str(job["method"]),
        "seed": int(job["train_seed"]),
        "cell_id": str(job["cell_id"]),
        "replicate_index": int(job["replicate_index"]),
        "study": str(job["study"]),
        "parent_cell_id": str(job["parent_cell_id"]),
        "supervision_token_cap": job.get("supervision_token_cap"),
        "token_cap_policy": job.get("token_cap_policy"),
    }


def identifiable_registry(*, repo_root: Path, config_path: Path) -> dict[str, Any]:
    extension, _, _ = _parent_contract(
        repo_root=repo_root, config_path=config_path.resolve()
    )
    output_root = repository_path(repo_root, str(extension["output_root"]))
    jobs: list[dict[str, Any]] = []
    for job in extension["job_order"]:
        matches: list[Path] = []
        if output_root.is_dir():
            for manifest_path in output_root.glob("*/manifest.json"):
                manifest = read_json_object(manifest_path)
                if manifest.get("config", {}).get("cell_id") == job["cell_id"]:
                    matches.append(manifest_path.parent)
        if len(matches) > 1:
            status = "ERROR_DUPLICATE_RUNS"
        elif not matches:
            status = "PENDING"
        else:
            formal = matches[0] / "audit" / "formal_cell_audit.json"
            ood = matches[0] / "audit" / "ood_audit.json"
            if formal.is_file() and ood.is_file():
                formal_status = read_json_object(formal).get("status")
                ood_status = read_json_object(ood).get("status")
                status = (
                    "AUDITED_PASS"
                    if formal_status == ood_status == "PASS"
                    else "AUDIT_FAILED"
                )
            elif (matches[0] / "cell_complete.json").is_file():
                status = "COMPLETE_UNAUDITED"
            else:
                status = "IN_PROGRESS"
        jobs.append(job | {"status": status, "run_dirs": [str(p) for p in matches]})
    return {
        "registry_version": "identifiable-budget-v4-registry-v1",
        "config_sha256": file_sha256(config_path.resolve()),
        "job_count": len(jobs),
        "audited_pass_count": sum(row["status"] == "AUDITED_PASS" for row in jobs),
        "jobs": jobs,
        "accuracy_withheld": True,
    }
