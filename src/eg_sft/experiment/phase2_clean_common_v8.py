"""Clean 24-cell common-mix block run entirely on the new GPU environment."""

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


MATRIX_VERSION = "phase2-clean-common24-v8"
PROTOCOL_ID = "phase2_clean_common24_v8"
METHODS = ("random_common_mix", "rds_error_common_mix")
TRAIN_SEEDS = (17, 29, 41)
REPLICATES = (1, 2, 3, 4)


def _selector(method: str) -> str:
    return "random" if method == "random_common_mix" else "rds_error"


def validate_clean_common_matrix(payload: dict[str, Any]) -> None:
    if payload.get("matrix_version") != MATRIX_VERSION:
        raise ValueError("unexpected clean-common matrix version")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected clean-common protocol ID")
    if (
        int(payload.get("historical_external_cells", -1)) != 8
        or int(payload.get("new_clean_cells", -1)) != 24
        or payload.get("historical_cells_in_primary_analysis") is not False
    ):
        raise ValueError("clean-common block counts or historical boundary changed")
    jobs = payload.get("job_order")
    if not isinstance(jobs, list) or len(jobs) != 24:
        raise ValueError("clean-common block must contain exactly 24 jobs")
    ids = [str(row.get("cell_id", "")) for row in jobs]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("clean-common cell IDs must be present and unique")
    observed = {
        (str(row["method"]), int(row["replicate_index"]), int(row["train_seed"]))
        for row in jobs
    }
    expected = {
        (method, replicate, seed)
        for method in METHODS
        for replicate in REPLICATES
        for seed in TRAIN_SEEDS
    }
    if observed != expected:
        raise ValueError("clean-common factor crossing changed")
    for row in jobs:
        method = str(row["method"])
        replicate = int(row["replicate_index"])
        seed = int(row["train_seed"])
        if (
            row.get("cell_id") != f"v8_rep{replicate}_{method}_train{seed}"
            or row.get("parent_cell_id") != f"rep{replicate}_{method}_train17"
            or row.get("study") != "clean_new_environment_common_block"
            or row.get("selector") != _selector(method)
            or row.get("mix") != "common_mix"
            or len(str(row.get("parent_selection_manifest_sha256", ""))) != 64
        ):
            raise ValueError(f"clean-common job binding changed: {row.get('cell_id')}")
    policy = payload.get("execution_policy", {})
    required_policy = {
        "accuracy_blind_until_all_audits": True,
        "one_cell_per_gpu": True,
        "one_gpu_process_at_a_time": True,
        "ddp_forbidden": True,
        "formal_and_ood_audit_required": True,
        "automatic_unblinding": False,
        "required_new_audited_cells": 24,
        "historical_seed17_external_only": True,
        "free_mix_not_in_this_block": True,
    }
    for field, expected_value in required_policy.items():
        if policy.get(field) != expected_value:
            raise ValueError(f"clean-common execution policy changed: {field}")
    waves = payload.get("dual_gpu_schedule")
    if not isinstance(waves, list) or len(waves) != 12:
        raise ValueError("clean-common dual schedule must contain 12 waves")
    gpu0 = [str(row.get("gpu0", "")) for row in waves]
    gpu1 = [str(row.get("gpu1", "")) for row in waves]
    if (
        len(set(gpu0)) != 12
        or len(set(gpu1)) != 12
        or set(gpu0) & set(gpu1)
        or set(gpu0) | set(gpu1) != set(ids)
    ):
        raise ValueError("clean-common schedules overlap or omit cells")
    by_id = {str(row["cell_id"]): row for row in jobs}
    for wave_index, wave in enumerate(waves, start=1):
        left = by_id[str(wave["gpu0"])]
        right = by_id[str(wave["gpu1"])]
        if (
            int(wave.get("wave", -1)) != wave_index
            or int(left["replicate_index"]) != int(right["replicate_index"])
            or int(left["train_seed"]) != int(right["train_seed"])
            or {left["selector"], right["selector"]} != {"random", "rds_error"}
        ):
            raise ValueError("clean-common wave pairing changed")
    for worker_cells in (gpu0, gpu1):
        worker_jobs = [by_id[cell] for cell in worker_cells]
        selector_counts = {
            level: sum(row["selector"] == level for row in worker_jobs)
            for level in ("random", "rds_error")
        }
        seed_counts = {
            seed: sum(int(row["train_seed"]) == seed for row in worker_jobs)
            for seed in TRAIN_SEEDS
        }
        replicate_counts = {
            rep: sum(int(row["replicate_index"]) == rep for row in worker_jobs)
            for rep in REPLICATES
        }
        if (
            set(selector_counts.values()) != {6}
            or set(seed_counts.values()) != {4}
            or set(replicate_counts.values()) != {3}
        ):
            raise ValueError("clean-common GPU schedule is unbalanced")
    if "single_gpu_order" in payload:
        raise ValueError("v8 release supports only two independent single-GPU workers")


def _parent_contract(
    *, repo_root: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    extension = read_json_object(config_path.resolve())
    validate_clean_common_matrix(extension)
    parent_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=extension["parent_matrix"],
        label="clean-common parent matrix",
    )
    parent = read_json_object(parent_path)
    validate_matrix_config(parent)
    for label, binding in extension["runtime_contracts"].items():
        resolve_frozen_file(
            repo_root=repo_root,
            binding=binding,
            label=f"clean-common {label} contract",
        )
    return extension, parent


def materialize_runtime_matrix(
    *, repo_root: Path, config_path: Path
) -> dict[str, Any]:
    extension, parent = _parent_contract(repo_root=repo_root, config_path=config_path)
    runtime = deepcopy(parent)
    runtime["phase1_protocol_version"] = MATRIX_VERSION
    runtime["matrix_version"] = MATRIX_VERSION
    runtime["output_root"] = str(extension["output_root"])
    runtime["job_order"] = deepcopy(extension["job_order"])
    runtime["execution_policy"] = {
        **runtime.get("execution_policy", {}),
        "required_audited_cells_before_unblinding": 24,
        "accuracy_blind_until_all_audits": True,
        "automatic_unblinding": False,
        "ood_audits_required_before_unblinding": True,
        "one_gpu_process_at_a_time": True,
    }
    runtime["phase2_extension"] = {
        "matrix_version": MATRIX_VERSION,
        "extension_config_sha256": file_sha256(config_path.resolve()),
        "historical_seed17_external_only": True,
    }
    return runtime


def resolve_clean_common_contract(
    *, repo_root: Path, config_path: Path, cell_id: str
) -> dict[str, Any]:
    config_path = config_path.resolve()
    extension, parent = _parent_contract(repo_root=repo_root, config_path=config_path)
    matching = [row for row in extension["job_order"] if row["cell_id"] == cell_id]
    if len(matching) != 1:
        raise ValueError("requested clean-common cell is absent or duplicated")
    job = matching[0]
    parent_matching = [
        row for row in parent["job_order"] if row["cell_id"] == job["parent_cell_id"]
    ]
    if len(parent_matching) != 1:
        raise ValueError("clean-common parent binding is absent or duplicated")
    parent_job = parent_matching[0]
    if (
        str(parent_job["method"]) != str(job["method"])
        or int(parent_job["replicate_index"]) != int(job["replicate_index"])
        or str(parent_job["selection_manifest"]["sha256"])
        != str(job["parent_selection_manifest_sha256"])
    ):
        raise ValueError("clean-common job differs from its frozen parent selection")
    runtime = materialize_runtime_matrix(repo_root=repo_root, config_path=config_path)
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
        raise ValueError("targeted-policy gate does not permit clean-common block")
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
        "study": "clean_new_environment_common_block",
        "parent_cell_id": str(job["parent_cell_id"]),
        "supervision_token_cap": None,
        "token_cap_policy": None,
    }


def clean_common_registry(
    *, repo_root: Path, config_path: Path
) -> dict[str, Any]:
    extension, _ = _parent_contract(repo_root=repo_root, config_path=config_path)
    output_root = repository_path(repo_root, str(extension["output_root"]))
    jobs = []
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
        jobs.append(job | {"status": status, "run_dirs": [str(path) for path in matches]})
    return {
        "registry_version": "phase2-clean-common24-v8-registry-v1",
        "config_sha256": file_sha256(config_path.resolve()),
        "job_count": len(jobs),
        "audited_pass_count": sum(row["status"] == "AUDITED_PASS" for row in jobs),
        "jobs": jobs,
        "historical_seed17_external_only": True,
        "accuracy_withheld": True,
    }
