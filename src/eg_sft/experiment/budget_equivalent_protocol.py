"""Configuration validation and CPU preflight for budget-equivalent v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eg_sft.selection.budget_equivalent import CORE_METHODS
from eg_sft.training.b500 import file_sha256


PROTOCOL_VERSION = "budget-equivalent-selection-v3"


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def repository_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("protocol paths must be repository-relative")
    resolved = (repo_root / relative).resolve()
    resolved.relative_to(repo_root.resolve())
    return resolved


def validate_protocol_config(payload: dict[str, Any]) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unexpected budget-equivalent protocol version")
    if tuple(payload.get("methods", ())) != CORE_METHODS:
        raise ValueError("core method order changed")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection contract is missing")
    fixed = {
        "selected_example_count": 500,
        "target_response_supervision_tokens": 32000,
        "response_tolerance_fraction": 0.005,
        "phase1_train_seed": 17,
    }
    for key, expected in fixed.items():
        if selection.get(key) != expected:
            raise ValueError(f"frozen selection field changed: {key}")
    rds = tuple(int(seed) for seed in selection.get("selection_replicate_seeds", ()))
    random_seeds = tuple(int(seed) for seed in selection.get("random_priority_seeds", ()))
    if rds != (101, 202, 303, 404):
        raise ValueError("phase1 RDS replicate seeds changed")
    if random_seeds != (1101, 1202, 1303, 1404):
        raise ValueError("phase1 random seeds changed")
    if not str(payload.get("output_root", "")).startswith(".aris/"):
        raise ValueError("runtime output must remain below ignored .aris")


def phase1_jobs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    validate_protocol_config(payload)
    selection = payload["selection"]
    rds_seeds = selection["selection_replicate_seeds"]
    random_seeds = selection["random_priority_seeds"]
    jobs = []
    for replicate_index, (rds_seed, random_seed) in enumerate(
        zip(rds_seeds, random_seeds, strict=True), start=1
    ):
        for method in CORE_METHODS:
            selection_seed = random_seed if method.startswith("random") else rds_seed
            jobs.append(
                {
                    "cell_id": f"rep{replicate_index}_{method}_train17",
                    "replicate_index": replicate_index,
                    "method": method,
                    "selection_seed": selection_seed,
                    "train_seed": 17,
                }
            )
    return jobs


def _binding_status(repo_root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = repository_path(repo_root, str(binding["path"]))
    expected = binding.get("sha256")
    if not path.is_file():
        return {"status": "BLOCKED_MISSING", "path": str(path), "sha256": None}
    observed = file_sha256(path)
    if not isinstance(expected, str) or len(expected) != 64:
        return {"status": "BLOCKED_UNFROZEN_SHA256", "path": str(path), "sha256": observed}
    if observed != expected:
        return {"status": "BLOCKED_HASH_MISMATCH", "path": str(path), "sha256": observed}
    return {"status": "READY", "path": str(path), "sha256": observed}


def preflight_protocol(*, repo_root: Path, config_path: Path) -> dict[str, Any]:
    payload = read_json_object(config_path)
    validate_protocol_config(payload)
    bindings = {
        name: _binding_status(repo_root, payload[name])
        for name in (
            "protocol_config",
            "candidate_inventory",
            "query_inventory",
            "similarity_artifact",
            "near_duplicate_clusters",
        )
    }
    ready = all(row["status"] == "READY" for row in bindings.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "READY" if ready else "BLOCKED",
        "bindings": bindings,
        "phase1_job_count": len(phase1_jobs(payload)),
        "phase1_jobs": phase1_jobs(payload),
        "formal_selection_permitted": ready,
        "claim_boundary": (
            "READY only validates frozen Phase 0 inputs; it is not an experiment result."
        ),
    }
