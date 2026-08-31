"""Frozen contract, dry-run, and accuracy-blind registry for cloud-v2 B=500."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eg_sft.experiment.cloud_v2_calibration import repository_path
from eg_sft.experiment.run_manifest import stable_config_hash
from eg_sft.training.b500 import (
    file_sha256,
    selected_id_sha256,
    validate_selection_manifest,
)
from eg_sft.training.effective_batch import validate_micro_batch_contract


FORMAL_VERSION = "cloud-v2-formal-b500-single-cell-v1"
FORMAL_METHODS = ("random", "rds_all", "rds_error")
FORMAL_SEEDS = (17, 29, 41)


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def resolve_frozen_file(
    *,
    repo_root: Path,
    binding: dict[str, Any],
    label: str,
) -> Path:
    path = repository_path(repo_root, str(binding["path"]), label=label)
    expected = binding.get("sha256")
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} SHA-256 is not frozen")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 changed: {observed} != {expected}")
    return path


def validate_formal_config(payload: dict[str, Any]) -> None:
    if payload.get("formal_protocol_version") != FORMAL_VERSION:
        raise ValueError("unexpected cloud-v2 formal protocol version")
    if tuple(payload.get("methods", ())) != FORMAL_METHODS:
        raise ValueError("formal methods changed")
    if tuple(int(seed) for seed in payload.get("seeds", ())) != FORMAL_SEEDS:
        raise ValueError("formal seeds changed")
    observed_jobs = [
        (str(row["method"]), int(row["seed"])) for row in payload.get("job_order", [])
    ]
    expected_jobs = {
        (method, seed) for method in FORMAL_METHODS for seed in FORMAL_SEEDS
    }
    if len(observed_jobs) != 9 or set(observed_jobs) != expected_jobs:
        raise ValueError("job_order must contain each formal cell exactly once")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("training config is missing")
    fixed_training = {
        "selection_budget": 500,
        "epochs": 2,
        "max_length": 512,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "nominal_effective_batch_size": 16,
        "loss_normalization": "effective_batch_response_token_sum_over_count",
        "checkpoint_every_optimizer_steps": 10,
        "single_training_process": True,
    }
    for key, value in fixed_training.items():
        if training.get(key) != value:
            raise ValueError(f"formal training field changed: {key}")
    validate_micro_batch_contract(
        micro_batch_size=int(training["micro_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        nominal_effective_batch_size=int(training["nominal_effective_batch_size"]),
    )
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation config is missing")
    if int(evaluation.get("expected_record_count", 0)) != 1319:
        raise ValueError("formal evaluation must contain 1319 records")
    if int(evaluation.get("worker_count", 0)) != 2:
        raise ValueError("formal evaluation requires exactly two workers")
    if int(evaluation.get("physical_batch_size_per_worker", 0)) != 1:
        raise ValueError("formal evaluation workers must keep batch size one")
    if evaluation.get("forbid_batch_size_above_one") is not True:
        raise ValueError("formal evaluation must explicitly forbid batch size above one")
    raw_shards = evaluation.get("shards")
    expected_shards = (
        ("test_shard0", 0, 660),
        ("test_shard1", 660, 1319),
    )
    observed_shards = tuple(
        (str(row["shard_id"]), int(row["start_index"]), int(row["end_index"]))
        for row in raw_shards or []
    )
    if observed_shards != expected_shards:
        raise ValueError("formal test shards must remain [0,660) and [660,1319)")
    policy = payload.get("execution_policy")
    if not isinstance(policy, dict):
        raise ValueError("execution policy is missing")
    if policy.get("one_cell_per_invocation") is not True:
        raise ValueError("formal runner must accept exactly one cell")
    if policy.get("automatic_next_cell") is not False:
        raise ValueError("formal runner must never auto-start the next cell")
    if policy.get("stdout_withholds_accuracy_and_method_comparison") is not True:
        raise ValueError("formal stdout must remain accuracy-blind")
    output_root = str(payload.get("output_root", ""))
    if not output_root.startswith(".aris/"):
        raise ValueError("formal runtime output must remain under the ignored .aris directory")


def _validate_data_files(*, repo_root: Path, config: dict[str, Any]) -> Path:
    spec = config["data_manifest"]
    directory = repository_path(repo_root, str(spec["directory"]), label="data manifest")
    for filename, expected in spec["required_files"].items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"frozen data file changed: {filename}")
    return directory


def _validate_selection(
    *,
    repo_root: Path,
    config: dict[str, Any],
    method: str,
    selection_seed: int,
) -> dict[str, Any]:
    path = resolve_frozen_file(
        repo_root=repo_root,
        binding=config["selections"][method],
        label=f"{method} selection",
    )
    manifest = read_json_object(path)
    selected = validate_selection_manifest(
        manifest,
        expected_strategy=method,
        expected_budget=500,
        expected_selection_seed=selection_seed,
    )
    observed_id_hash = selected_id_sha256(selected)
    if observed_id_hash != manifest.get("selected_id_sha256"):
        raise ValueError(f"{method} selected ID SHA-256 changed")
    return {
        "path": path,
        "manifest": manifest,
        "selected": selected,
        "file_sha256": file_sha256(path),
        "selected_id_sha256": observed_id_hash,
    }


def resolve_formal_contract(
    *,
    repo_root: Path,
    config_path: Path,
    method: str,
    seed: int,
) -> dict[str, Any]:
    config = read_json_object(config_path)
    validate_formal_config(config)
    if method not in FORMAL_METHODS or seed not in FORMAL_SEEDS:
        raise ValueError("requested method/seed is outside the frozen matrix")
    protocol_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=config["protocol_config"],
        label="protocol config",
    )
    recipe_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=config["base_recipe_config"],
        label="base recipe config",
    )
    data_dir = _validate_data_files(repo_root=repo_root, config=config)
    gate_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=config["h1a_gate"],
        label="H1a gate",
    )
    gate = read_json_object(gate_path)
    if gate.get(config["h1a_gate"]["required_field"]) != config["h1a_gate"][
        "required_value"
    ]:
        raise ValueError("frozen H1a gate does not permit formal B=500")
    base_recipe = read_json_object(recipe_path)
    selection = _validate_selection(
        repo_root=repo_root,
        config=config,
        method=method,
        selection_seed=int(base_recipe["selection"]["selection_seed"]),
    )
    output_root = repository_path(
        repo_root,
        str(config["output_root"]),
        label="formal output root",
    )
    return {
        "config": config,
        "config_path": config_path,
        "config_sha256": file_sha256(config_path),
        "protocol_path": protocol_path,
        "protocol": read_json_object(protocol_path),
        "base_recipe_path": recipe_path,
        "base_recipe": base_recipe,
        "data_dir": data_dir,
        "gate_path": gate_path,
        "selection": selection,
        "output_root": output_root,
        "method": method,
        "seed": seed,
    }


def _cell_registry_status(run_dirs: list[Path]) -> dict[str, Any]:
    if not run_dirs:
        return {"status": "PENDING", "run_dir": None}
    if len(run_dirs) != 1:
        return {
            "status": "ERROR_DUPLICATE_RUNS",
            "run_dirs": [str(path) for path in run_dirs],
        }
    run_dir = run_dirs[0]
    completion = run_dir / "cell_complete.json"
    audit = run_dir / "audit" / "formal_cell_audit.json"
    if audit.is_file() and read_json_object(audit).get("status") == "PASS":
        status = "AUDITED_PASS"
    elif completion.is_file():
        status = "COMPLETE_UNAUDITED"
    else:
        status = "IN_PROGRESS"
    return {"status": status, "run_dir": str(run_dir)}


def build_formal_registry(
    *,
    repo_root: Path,
    config_path: Path,
    python_executable: str = "python",
) -> dict[str, Any]:
    config = read_json_object(config_path)
    validate_formal_config(config)
    base_recipe_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=config["base_recipe_config"],
        label="base recipe config",
    )
    base_recipe = read_json_object(base_recipe_path)
    _validate_data_files(repo_root=repo_root, config=config)
    for method in FORMAL_METHODS:
        _validate_selection(
            repo_root=repo_root,
            config=config,
            method=method,
            selection_seed=int(base_recipe["selection"]["selection_seed"]),
        )
    output_root = repository_path(
        repo_root,
        str(config["output_root"]),
        label="formal output root",
    )
    manifests = list(output_root.glob("*/manifest.json")) if output_root.is_dir() else []
    indexed: dict[tuple[str, int], list[Path]] = {
        (method, seed): [] for method in FORMAL_METHODS for seed in FORMAL_SEEDS
    }
    for manifest_path in manifests:
        manifest = read_json_object(manifest_path)
        run_config = manifest.get("config", {})
        key = (str(run_config.get("method", "")), int(manifest.get("seed", -1)))
        if key in indexed:
            indexed[key].append(manifest_path.parent)
    jobs = []
    relative_config = config_path.resolve().relative_to(repo_root.resolve()).as_posix()
    for order, job in enumerate(config["job_order"], start=1):
        method = str(job["method"])
        seed = int(job["seed"])
        registry = _cell_registry_status(indexed[(method, seed)])
        jobs.append(
            {
                "order": order,
                "cell_id": f"{method}_seed_{seed}",
                "method": method,
                "seed": seed,
                **registry,
                "command": [
                    python_executable,
                    "scripts/run_cloud_v2_formal_cell.py",
                    "--config",
                    relative_config,
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                ],
            }
        )
    return {
        "registry_version": "cloud-v2-formal-b500-registry-v1",
        "formal_config_sha256": file_sha256(config_path),
        "registry_config_hash": stable_config_hash(config),
        "automatic_execution": False,
        "job_count": 9,
        "jobs": jobs,
        "accuracy_withheld": True,
        "claim_boundary": "Engineering registry only; no method comparison is exposed.",
    }


def engineering_stdout_payload(
    *,
    status: str,
    run_id: str | None,
    hashes: dict[str, str],
    stage: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "stage": stage,
        "run_id": run_id,
        "hashes": hashes,
        "accuracy_withheld": True,
        "method_comparison_withheld": True,
        "next_cell_started": False,
    }
