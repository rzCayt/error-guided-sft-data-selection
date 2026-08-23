"""Validation and provenance helpers for the isolated cloud-v2 calibration layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eg_sft.experiment.run_manifest import stable_config_hash
from eg_sft.training.b500 import file_sha256
from eg_sft.training.effective_batch import validate_micro_batch_contract


CALIBRATION_VERSION = "b500-cloud-v2-calibration-v1"
REQUIRED_PROFILE_PAIRS = {
    "mb1_ga16": (1, 16),
    "mb2_ga8": (2, 8),
    "mb4_ga4": (4, 4),
    "mb8_ga2": (8, 2),
}


@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    micro_batch_size: int
    gradient_accumulation_steps: int
    nominal_effective_batch_size: int


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def repository_path(repo_root: Path, value: str, *, label: str) -> Path:
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


def resolve_frozen_artifact(
    *,
    repo_root: Path,
    binding: dict[str, Any],
    label: str,
) -> Path:
    path = repository_path(repo_root, str(binding["path"]), label=label)
    expected = binding.get("sha256")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {binding['path']}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} must have a frozen SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 changed: observed {observed}, expected {expected}")
    return path


def validate_calibration_config(payload: dict[str, Any]) -> dict[str, CalibrationProfile]:
    if payload.get("calibration_version") != CALIBRATION_VERSION:
        raise ValueError("unexpected cloud-v2 calibration version")
    if int(payload.get("training_example_count", 0)) != 64:
        raise ValueError("cloud-v2 training calibration must use exactly 64 examples")
    if int(payload.get("generation_example_count", 0)) != 128:
        raise ValueError("cloud-v2 generation calibration must use exactly 128 examples")
    if int(payload.get("training_seed", -1)) != 17:
        raise ValueError("cloud-v2 calibration training seed must remain 17")
    if payload.get("selection_strategy") != "random":
        raise ValueError("cloud-v2 calibration must use the frozen random selection")
    if payload.get("attention_implementation") not in {"sdpa", "eager"}:
        raise ValueError("attention_implementation must be sdpa or eager")
    if not isinstance(payload.get("gradient_checkpointing"), bool):
        raise ValueError("gradient_checkpointing must be boolean")
    if payload.get("loss_normalization") != "effective_batch_response_token_sum_over_count":
        raise ValueError("cloud-v2 requires effective-batch response-token normalization")
    if int(payload.get("checkpoint_every_optimizer_steps", 0)) <= 0:
        raise ValueError("checkpoint interval must be positive")
    if payload.get("generation_protocol_split") != "development":
        raise ValueError("generation calibration must not tune on the held-out test split")
    batch_sizes = [int(value) for value in payload.get("generation_batch_sizes", [])]
    if batch_sizes != [1, 4, 8, 16]:
        raise ValueError("generation batch sizes must be frozen as [1, 4, 8, 16]")

    raw_profiles = payload.get("training_profiles")
    if not isinstance(raw_profiles, dict) or set(raw_profiles) != set(REQUIRED_PROFILE_PAIRS):
        raise ValueError("training_profiles must contain the four frozen calibration pairs")
    profiles: dict[str, CalibrationProfile] = {}
    for name, expected_pair in REQUIRED_PROFILE_PAIRS.items():
        row = raw_profiles[name]
        if not isinstance(row, dict):
            raise ValueError(f"training profile {name} must be an object")
        micro_batch_size = int(row["micro_batch_size"])
        accumulation = int(row["gradient_accumulation_steps"])
        effective = int(row["nominal_effective_batch_size"])
        if (micro_batch_size, accumulation) != expected_pair:
            raise ValueError(f"training profile {name} changed its frozen pair")
        validate_micro_batch_contract(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=accumulation,
            nominal_effective_batch_size=effective,
        )
        profiles[name] = CalibrationProfile(
            name=name,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=accumulation,
            nominal_effective_batch_size=effective,
        )
    return profiles


def calibration_config_hash(payload: dict[str, Any]) -> str:
    """Hash the complete calibration decision contract."""

    validate_calibration_config(payload)
    return stable_config_hash(payload)


def calibration_run_config(
    *,
    payload: dict[str, Any],
    profile: CalibrationProfile | None,
    generation_batch_size: int | None,
    adapter_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the exact manifest payload for one isolated calibration run."""

    validate_calibration_config(payload)
    if (profile is None) == (generation_batch_size is None):
        raise ValueError("select exactly one training profile or generation batch size")
    if generation_batch_size is not None and generation_batch_size not in payload[
        "generation_batch_sizes"
    ]:
        raise ValueError("generation batch size is outside the frozen calibration grid")
    mode = "training" if profile is not None else "generation"
    return {
        "study_role": "engineering_calibration_only_excluded_from_formal_matrix",
        "calibration_version": CALIBRATION_VERSION,
        "calibration_config_hash": calibration_config_hash(payload),
        "mode": mode,
        "training_profile": (
            {
                "name": profile.name,
                "micro_batch_size": profile.micro_batch_size,
                "gradient_accumulation_steps": profile.gradient_accumulation_steps,
                "nominal_effective_batch_size": profile.nominal_effective_batch_size,
            }
            if profile is not None
            else None
        ),
        "generation_batch_size": generation_batch_size,
        "adapter_sha256": adapter_sha256,
        "attention_implementation": payload["attention_implementation"],
        "gradient_checkpointing": payload["gradient_checkpointing"],
        "loss_normalization": payload["loss_normalization"],
    }
