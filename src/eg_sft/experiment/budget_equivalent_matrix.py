"""Frozen Phase 1 matrix contract for budget-equivalent v3 cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eg_sft.experiment.budget_equivalent_protocol import repository_path
from eg_sft.selection.budget_equivalent import CORE_METHODS, canonical_json_sha256
from eg_sft.training.b500 import (
    file_sha256,
    selected_id_sha256,
    validate_selection_manifest,
)


MATRIX_VERSION = "budget-equivalent-phase1-matrix-v3"


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def resolve_frozen_file(
    *, repo_root: Path, binding: dict[str, Any], label: str
) -> Path:
    path = repository_path(repo_root, str(binding["path"]))
    expected = binding.get("sha256")
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} SHA-256 is not frozen")
    if file_sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 changed")
    return path


def validate_matrix_config(payload: dict[str, Any]) -> None:
    if payload.get("phase1_protocol_version") != MATRIX_VERSION:
        raise ValueError("unexpected Phase 1 matrix version")
    if tuple(payload.get("methods", ())) != CORE_METHODS:
        raise ValueError("Phase 1 methods changed")
    jobs = payload.get("job_order")
    if not isinstance(jobs, list) or len(jobs) != 16:
        raise ValueError("Phase 1 matrix must contain exactly 16 jobs")
    observed = {
        (int(row["replicate_index"]), str(row["method"]), int(row["train_seed"]))
        for row in jobs
    }
    expected = {
        (replicate, method, 17)
        for replicate in range(1, 5)
        for method in CORE_METHODS
    }
    if observed != expected or len(observed) != 16:
        raise ValueError("Phase 1 job identities changed")
    training = payload.get("training", {})
    fixed_training = {
        "selection_budget": 500,
        "epochs": 2,
        "optimizer_steps": 64,
        "max_length": 512,
        "loss_normalization": "optimizer_step_response_token_sum_over_count",
        "single_training_process": True,
    }
    for key, expected_value in fixed_training.items():
        if training.get(key) != expected_value:
            raise ValueError(f"Phase 1 training field changed: {key}")
    evaluation = payload.get("evaluation", {})
    if int(evaluation.get("expected_record_count", 0)) != 1319:
        raise ValueError("Phase 1 primary evaluation must contain 1319 GSM8K rows")
    policy = payload.get("execution_policy", {})
    if policy.get("one_cell_per_invocation") is not True:
        raise ValueError("Phase 1 runner must execute one cell per invocation")
    if policy.get("automatic_next_cell") is not False:
        raise ValueError("Phase 1 runner must not auto-start another cell")
    if policy.get("accuracy_blind_until_all_audits") is not True:
        raise ValueError("Phase 1 intermediate output must remain accuracy-blind")
    if not str(payload.get("output_root", "")).startswith(".aris/"):
        raise ValueError("Phase 1 outputs must remain below .aris")


def _validate_selection(
    *, repo_root: Path, job: dict[str, Any]
) -> dict[str, Any]:
    binding = job["selection_manifest"]
    path = resolve_frozen_file(
        repo_root=repo_root,
        binding=binding,
        label=f"{job['cell_id']} selection",
    )
    manifest = read_json_object(path)
    claimed_content_hash = manifest.get("manifest_content_sha256")
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    if claimed_content_hash != canonical_json_sha256(content):
        raise ValueError("selection manifest self-hash changed")
    selected = validate_selection_manifest(
        manifest,
        expected_strategy=str(job["method"]),
        expected_budget=500,
        expected_selection_seed=int(job["selection_seed"]),
    )
    if selected_id_sha256(selected) != manifest.get("selected_id_sha256"):
        raise ValueError("selected ID hash changed")
    audit = manifest.get("budget_audit", {})
    if int(audit.get("selected_count", -1)) != 500:
        raise ValueError("selection count audit failed")
    if float(audit.get("response_relative_error", 1.0)) > 0.005:
        raise ValueError("response supervision token budget audit failed")
    if audit.get("duplicate_cluster_mode") != "near_duplicate_cluster_manifest":
        raise ValueError("formal matrix forbids exact-prompt duplicate fallback")
    if str(job["method"]).endswith("common_mix"):
        if audit.get("common_mix_quota_matches") is not True:
            raise ValueError("common-mix quotas do not match")
        if float(audit.get("prompt_relative_error", 1.0)) > 0.01:
            raise ValueError("common-mix prompt token audit failed")
        if float(audit.get("total_relative_error", 1.0)) > 0.01:
            raise ValueError("common-mix total token audit failed")
    return {
        "path": path,
        "manifest": manifest,
        "selected": selected,
        "file_sha256": file_sha256(path),
        "selected_id_sha256": manifest["selected_id_sha256"],
    }


def resolve_phase1_contract(
    *,
    repo_root: Path,
    config_path: Path,
    cell_id: str,
) -> dict[str, Any]:
    config = read_json_object(config_path)
    if config.get("matrix_version") == "phase2-clean-common24-v8":
        from eg_sft.experiment.phase2_clean_common_v8 import (
            resolve_clean_common_contract,
        )

        return resolve_clean_common_contract(
            repo_root=repo_root,
            config_path=config_path,
            cell_id=cell_id,
        )
    if config.get("matrix_version") == "phase2-crossed-48cell-v7":
        from eg_sft.experiment.phase2_crossed_v7 import resolve_phase2_contract

        return resolve_phase2_contract(
            repo_root=repo_root,
            config_path=config_path,
            cell_id=cell_id,
        )
    if config.get("matrix_version") == "identifiable-budget-v4-extension-v1":
        from eg_sft.experiment.identifiable_budget_v4 import (
            resolve_identifiable_contract,
        )

        return resolve_identifiable_contract(
            repo_root=repo_root,
            config_path=config_path,
            cell_id=cell_id,
        )
    validate_matrix_config(config)
    matching = [row for row in config["job_order"] if row["cell_id"] == cell_id]
    if len(matching) != 1:
        raise ValueError("requested cell is absent or duplicated")
    job = matching[0]
    protocol_path = resolve_frozen_file(
        repo_root=repo_root, binding=config["protocol_config"], label="protocol config"
    )
    recipe_path = resolve_frozen_file(
        repo_root=repo_root, binding=config["base_recipe_config"], label="base recipe"
    )
    gate_path = resolve_frozen_file(
        repo_root=repo_root, binding=config["information_gates"], label="information gates"
    )
    gates = read_json_object(gate_path)
    if gates.get("targeted_policy_gate_passed") is not True:
        raise ValueError("targeted-policy information gate does not permit Phase 1")
    if gates.get("formal_near_duplicate_control") is not True:
        raise ValueError("formal near-duplicate control did not pass")
    data_spec = config["data_manifest"]
    data_dir = repository_path(repo_root, str(data_spec["directory"]))
    for filename, expected_hash in data_spec["required_files"].items():
        path = data_dir / filename
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"frozen data file changed: {filename}")
    selection = _validate_selection(repo_root=repo_root, job=job)
    return {
        "config": config,
        "config_path": config_path,
        "config_sha256": file_sha256(config_path),
        "protocol_path": protocol_path,
        "protocol": read_json_object(protocol_path),
        "base_recipe_path": recipe_path,
        "base_recipe": read_json_object(recipe_path),
        "gate_path": gate_path,
        "data_dir": data_dir,
        "selection": selection,
        "output_root": repository_path(repo_root, str(config["output_root"])),
        "method": str(job["method"]),
        "seed": int(job["train_seed"]),
        "cell_id": cell_id,
        "replicate_index": int(job["replicate_index"]),
    }


def phase1_registry(*, repo_root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json_object(config_path)
    if config.get("matrix_version") == "phase2-clean-common24-v8":
        from eg_sft.experiment.phase2_clean_common_v8 import clean_common_registry

        return clean_common_registry(repo_root=repo_root, config_path=config_path)
    if config.get("matrix_version") == "phase2-crossed-48cell-v7":
        from eg_sft.experiment.phase2_crossed_v7 import phase2_registry

        return phase2_registry(repo_root=repo_root, config_path=config_path)
    if config.get("matrix_version") == "identifiable-budget-v4-extension-v1":
        from eg_sft.experiment.identifiable_budget_v4 import identifiable_registry

        return identifiable_registry(repo_root=repo_root, config_path=config_path)
    validate_matrix_config(config)
    output_root = repository_path(repo_root, str(config["output_root"]))
    require_ood_audit = bool(
        config.get("execution_policy", {}).get(
            "ood_audits_required_before_unblinding", False
        )
    )
    jobs = []
    for job in config["job_order"]:
        matches = []
        if output_root.is_dir():
            for manifest_path in output_root.glob("*/manifest.json"):
                manifest = read_json_object(manifest_path)
                if manifest.get("config", {}).get("cell_id") == job["cell_id"]:
                    matches.append(manifest_path.parent)
        if len(matches) > 1:
            status = "ERROR_DUPLICATE_RUNS"
        elif not matches:
            status = "PENDING"
        elif (matches[0] / "audit" / "formal_cell_audit.json").is_file():
            audit = read_json_object(matches[0] / "audit" / "formal_cell_audit.json")
            if audit.get("status") != "PASS":
                status = "AUDIT_FAILED"
            elif require_ood_audit:
                ood_path = matches[0] / "audit" / "ood_audit.json"
                if not ood_path.is_file():
                    status = "FORMAL_AUDITED_OOD_PENDING"
                else:
                    ood_audit = read_json_object(ood_path)
                    status = (
                        "AUDITED_PASS"
                        if ood_audit.get("status") == "PASS"
                        else "OOD_AUDIT_FAILED"
                    )
            else:
                status = "AUDITED_PASS"
        elif (matches[0] / "cell_complete.json").is_file():
            status = "COMPLETE_UNAUDITED"
        else:
            status = "IN_PROGRESS"
        jobs.append(job | {"status": status, "run_dirs": [str(path) for path in matches]})
    return {
        "registry_version": "budget-equivalent-phase1-registry-v3",
        "config_sha256": file_sha256(config_path),
        "job_count": len(jobs),
        "audited_pass_count": sum(row["status"] == "AUDITED_PASS" for row in jobs),
        "ood_audits_required_before_unblinding": require_ood_audit,
        "jobs": jobs,
        "accuracy_withheld": True,
    }
