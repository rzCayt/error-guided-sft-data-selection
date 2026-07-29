"""Preflight the frozen random/RDS B=500 comparison without running training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eg_sft.experiment.run_manifest import stable_config_hash
from eg_sft.training.b500 import (
    file_sha256,
    read_jsonl,
    selected_id_sha256,
    validate_selection_manifest,
)

FORMAL_STRATEGIES = ("random", "rds_all", "rds_error")
FORMAL_SEEDS = (17, 29, 41)
MATRIX_VERSION = "b500-formal-comparison-v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _repo_path(repo_root: Path, value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository root") from error
    return resolved


def _verify_frozen_file(
    *,
    repo_root: Path,
    binding: dict[str, Any],
    label: str,
) -> tuple[Path, str]:
    path = _repo_path(repo_root, str(binding["path"]), label=label)
    if not path.is_file():
        raise ValueError(f"{label} is missing: {binding['path']}")
    expected = binding.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} must have a frozen SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 changed: observed {observed}, expected {expected}"
        )
    return path, observed


def _validate_schedule(spec: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = spec.get("job_order")
    if not isinstance(schedule, list):
        raise ValueError("job_order must be a list")
    normalized = [
        {
            "strategy": str(row["strategy"]),
            "seed": int(row["seed"]),
        }
        for row in schedule
    ]
    expected = {
        (strategy, seed)
        for strategy in FORMAL_STRATEGIES
        for seed in FORMAL_SEEDS
    }
    observed = {(row["strategy"], row["seed"]) for row in normalized}
    if len(normalized) != len(expected) or observed != expected:
        raise ValueError("job_order must contain each strategy-seed pair exactly once")
    return normalized


def _candidate_pool_index(candidate_pool_path: Path) -> dict[str, dict[str, Any]]:
    candidates = read_jsonl(candidate_pool_path)
    indexed = {str(row["candidate_id"]): row for row in candidates}
    if len(indexed) != len(candidates):
        raise ValueError("candidate pool contains duplicate candidate IDs")
    return indexed


def _validate_selected_candidate(
    selected: dict[str, Any],
    frozen: dict[str, Any],
) -> None:
    fields = (
        "candidate_id",
        "source_index",
        "prompt_sha256",
        "response_sha256",
    )
    for field in fields:
        if selected.get(field) != frozen.get(field):
            raise ValueError(
                f"{selected.get('candidate_id')} changed frozen field {field}"
            )
    if int(selected.get("total_tokens", 0)) <= 0:
        raise ValueError(f"{selected['candidate_id']} has invalid total_tokens")
    if int(selected.get("supervised_tokens", 0)) <= 0:
        raise ValueError(
            f"{selected['candidate_id']} has no supervised response tokens"
        )


def _selection_preflight(
    *,
    repo_root: Path,
    strategy: str,
    binding: dict[str, Any],
    budget: int,
    selection_seed: int,
    candidate_pool: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    relative_path = str(binding["path"])
    path = _repo_path(
        repo_root,
        relative_path,
        label=f"{strategy} selection manifest",
    )
    expected_sha256 = binding.get("sha256")
    if not path.is_file():
        return {
            "strategy": strategy,
            "path": relative_path,
            "status": "BLOCKED_MISSING_SELECTION_MANIFEST",
            "expected_sha256": expected_sha256,
        }
    observed_sha256 = file_sha256(path)
    if expected_sha256 is None:
        return {
            "strategy": strategy,
            "path": relative_path,
            "status": "BLOCKED_UNFROZEN_SELECTION_SHA256",
            "observed_sha256": observed_sha256,
            "expected_sha256": None,
        }
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{strategy} selection SHA-256 is malformed")
    if observed_sha256 != expected_sha256:
        return {
            "strategy": strategy,
            "path": relative_path,
            "status": "ERROR_SELECTION_SHA256_MISMATCH",
            "observed_sha256": observed_sha256,
            "expected_sha256": expected_sha256,
        }

    try:
        manifest = _read_json(path)
        selected = validate_selection_manifest(
            manifest,
            expected_strategy=strategy,
            expected_budget=budget,
            expected_selection_seed=selection_seed,
        )
        observed_selected_id_sha256 = selected_id_sha256(selected)
        if observed_selected_id_sha256 != manifest.get("selected_id_sha256"):
            raise ValueError("selected candidate ID hash changed")
        for candidate in selected:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id not in candidate_pool:
                raise ValueError(f"{candidate_id} is absent from the frozen pool")
            _validate_selected_candidate(candidate, candidate_pool[candidate_id])
    except (KeyError, TypeError, ValueError) as error:
        return {
            "strategy": strategy,
            "path": relative_path,
            "status": "ERROR_INVALID_SELECTION_MANIFEST",
            "observed_sha256": observed_sha256,
            "error": str(error),
        }
    return {
        "strategy": strategy,
        "path": relative_path,
        "status": "READY",
        "observed_sha256": observed_sha256,
        "selected_count": len(selected),
        "selected_id_sha256": observed_selected_id_sha256,
    }


def preflight_b500_matrix(
    *,
    spec: dict[str, Any],
    repo_root: Path,
    python_executable: str = "python",
) -> dict[str, Any]:
    """Return a deterministic nine-job report; never launch a subprocess."""

    if spec.get("matrix_version") != MATRIX_VERSION:
        raise ValueError("matrix_version is not the frozen v1 protocol")
    if tuple(int(seed) for seed in spec["formal_training_seeds"]) != FORMAL_SEEDS:
        raise ValueError("formal training seeds changed")
    selections = spec.get("selections")
    if not isinstance(selections, dict) or tuple(selections) != FORMAL_STRATEGIES:
        raise ValueError("selection strategies or their order changed")
    execution_policy = spec.get("execution_policy")
    if not isinstance(execution_policy, dict):
        raise ValueError("execution_policy must be an object")
    if execution_policy.get("automatic_execution") is not False:
        raise ValueError("matrix v1 must not execute jobs automatically")
    if execution_policy.get("one_job_per_manual_invocation") is not True:
        raise ValueError("matrix v1 requires one manually monitored job at a time")

    protocol_path, protocol_sha256 = _verify_frozen_file(
        repo_root=repo_root,
        binding=spec["protocol_config"],
        label="protocol config",
    )
    recipe_path, recipe_sha256 = _verify_frozen_file(
        repo_root=repo_root,
        binding=spec["recipe_config"],
        label="recipe config",
    )
    runner_path, runner_sha256 = _verify_frozen_file(
        repo_root=repo_root,
        binding=spec["runner"],
        label="B500 runner",
    )
    recipe = _read_json(recipe_path)
    if tuple(recipe["selection"]["allowed_strategies"]) != FORMAL_STRATEGIES:
        raise ValueError("recipe strategies changed")
    if tuple(int(seed) for seed in recipe["formal_training_seeds"]) != FORMAL_SEEDS:
        raise ValueError("recipe training seeds changed")
    budget = int(recipe["selection"]["budget"])
    selection_seed = int(recipe["selection"]["selection_seed"])
    if budget != 500:
        raise ValueError("formal selection budget must remain 500")

    data_spec = spec["data_manifest"]
    data_directory = _repo_path(
        repo_root,
        str(data_spec["directory"]),
        label="data manifest directory",
    )
    data_files: dict[str, Any] = {}
    for filename, expected_sha256 in data_spec["required_files"].items():
        path = data_directory / filename
        if not path.is_file():
            raise ValueError(f"frozen data file is missing: {filename}")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"frozen data file SHA-256 changed: {filename}")
        data_files[filename] = {
            "path": f"{data_spec['directory']}/{filename}",
            "sha256": observed_sha256,
        }

    h1a_path, h1a_sha256 = _verify_frozen_file(
        repo_root=repo_root,
        binding=spec["h1a_gate"],
        label="H1a gate artifact",
    )
    h1a = _read_json(h1a_path)
    required_field = str(spec["h1a_gate"]["required_field"])
    required_value = spec["h1a_gate"]["required_value"]
    if h1a.get(required_field) != required_value:
        raise ValueError("the preregistered H1a gate does not permit B500 comparison")

    candidate_pool = _candidate_pool_index(
        data_directory / "tulu_candidate_pool.jsonl"
    )
    selection_reports = {
        strategy: _selection_preflight(
            repo_root=repo_root,
            strategy=strategy,
            binding=selections[strategy],
            budget=budget,
            selection_seed=selection_seed,
            candidate_pool=candidate_pool,
        )
        for strategy in FORMAL_STRATEGIES
    }
    all_selections_ready = all(
        report["status"] == "READY" for report in selection_reports.values()
    )

    schedule = _validate_schedule(spec)
    common_contract = {
        "matrix_version": MATRIX_VERSION,
        "protocol_config_sha256": protocol_sha256,
        "recipe_config_sha256": recipe_sha256,
        "runner_sha256": runner_sha256,
        "data_file_sha256": {
            name: report["sha256"] for name, report in data_files.items()
        },
        "h1a_gate_sha256": h1a_sha256,
        "budget": budget,
        "selection_seed": selection_seed,
    }
    common_contract_sha256 = stable_config_hash(common_contract)
    output_root = str(spec["output_root"])
    _repo_path(repo_root, output_root, label="output root")
    jobs: list[dict[str, Any]] = []
    for order_index, scheduled in enumerate(schedule, start=1):
        strategy = scheduled["strategy"]
        seed = scheduled["seed"]
        selection_report = selection_reports[strategy]
        if selection_report["status"] != "READY":
            status = selection_report["status"]
        elif not all_selections_ready:
            status = "BLOCKED_UNTIL_ALL_SELECTIONS_ARE_FROZEN"
        else:
            status = "READY_FOR_MANUAL_INVOCATION"
        command = [
            python_executable,
            str(spec["runner"]["path"]),
            "--protocol-config",
            str(spec["protocol_config"]["path"]),
            "--recipe-config",
            str(spec["recipe_config"]["path"]),
            "--selection-manifest",
            str(selections[strategy]["path"]),
            "--data-manifest-dir",
            str(data_spec["directory"]),
            "--output-root",
            output_root,
            "--strategy",
            strategy,
            "--seed",
            str(seed),
        ]
        jobs.append(
            {
                "order": order_index,
                "job_id": f"b500_{strategy}_seed_{seed}",
                "strategy": strategy,
                "seed": seed,
                "status": status,
                "selection_manifest": str(selections[strategy]["path"]),
                "common_contract_sha256": common_contract_sha256,
                "command": command,
            }
        )

    return {
        "matrix_version": MATRIX_VERSION,
        "matrix_config_sha256": stable_config_hash(spec),
        "status": (
            "READY_FOR_MANUAL_ONE_JOB_AT_A_TIME"
            if all_selections_ready
            else "BLOCKED_INCOMPLETE_SELECTION_FREEZE"
        ),
        "automatic_execution": False,
        "job_count": len(jobs),
        "ready_selection_count": sum(
            report["status"] == "READY"
            for report in selection_reports.values()
        ),
        "common_contract": common_contract,
        "common_contract_sha256": common_contract_sha256,
        "artifacts": {
            "protocol_config": {
                "path": str(protocol_path.relative_to(repo_root.resolve())),
                "sha256": protocol_sha256,
            },
            "recipe_config": {
                "path": str(recipe_path.relative_to(repo_root.resolve())),
                "sha256": recipe_sha256,
            },
            "runner": {
                "path": str(runner_path.relative_to(repo_root.resolve())),
                "sha256": runner_sha256,
            },
            "data_files": data_files,
            "h1a_gate": {
                "path": str(h1a_path.relative_to(repo_root.resolve())),
                "sha256": h1a_sha256,
                required_field: h1a[required_field],
            },
        },
        "selections": selection_reports,
        "jobs": jobs,
        "next_blockers": [
            {
                "strategy": strategy,
                "status": report["status"],
                "path": report["path"],
            }
            for strategy, report in selection_reports.items()
            if report["status"] != "READY"
        ],
        "claim_boundary": (
            "This is a dry-run preflight only. It does not train a model, "
            "evaluate GSM8K, or establish a selector comparison result."
        ),
    }
