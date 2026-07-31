"""Runtime safety helpers for one resumable formal B=500 job."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ThermalPolicy:
    """Validated temperature and pacing policy for local GPU work."""

    start_max_c: int
    pause_at_c: int
    resume_at_c: int
    hard_stop_at_c: int
    poll_seconds: float
    training_check_every_micro_batches: int
    training_inter_micro_batch_sleep_seconds: float
    checkpoint_every_optimizer_steps: int
    evaluation_check_every_examples: int
    evaluation_inter_example_sleep_seconds: float


def validate_execution_policy(payload: dict[str, Any]) -> ThermalPolicy:
    """Validate physical execution controls without changing model semantics."""

    if payload.get("execution_policy_version") != ("b500-formal-local-resumable-v1"):
        raise ValueError("unexpected formal execution policy version")
    temperature = payload.get("temperature")
    training = payload.get("training")
    evaluation = payload.get("evaluation")
    resources = payload.get("resources")
    if not all(
        isinstance(section, dict) for section in (temperature, training, evaluation, resources)
    ):
        raise ValueError("formal execution policy sections must be objects")

    start_max_c = int(temperature["start_max_c"])
    resume_at_c = int(temperature["resume_at_c"])
    pause_at_c = int(temperature["pause_at_c"])
    hard_stop_at_c = int(temperature["hard_stop_at_c"])
    if not (0 < resume_at_c <= start_max_c < pause_at_c < hard_stop_at_c):
        raise ValueError("temperature thresholds must satisfy resume <= start < pause < hard stop")
    if pause_at_c > 75:
        raise ValueError("formal work must start cooling no later than 75C")
    if hard_stop_at_c > 80:
        raise ValueError("formal work must hard-stop no later than 80C")
    if int(evaluation["physical_batch_size"]) != 1:
        raise ValueError("formal local evaluation physical batch must be 1")
    if int(resources["cpu_threads"]) > 2:
        raise ValueError("formal local execution is limited to two CPU threads")
    if float(resources["min_free_system_memory_gib"]) < 6:
        raise ValueError("minimum free system memory guard is too low")
    if float(resources["min_free_disk_gib"]) < 20:
        raise ValueError("minimum free disk guard is too low")

    policy = ThermalPolicy(
        start_max_c=start_max_c,
        pause_at_c=pause_at_c,
        resume_at_c=resume_at_c,
        hard_stop_at_c=hard_stop_at_c,
        poll_seconds=float(temperature["poll_seconds"]),
        training_check_every_micro_batches=int(training["temperature_check_every_micro_batches"]),
        training_inter_micro_batch_sleep_seconds=float(training["inter_micro_batch_sleep_seconds"]),
        checkpoint_every_optimizer_steps=int(training["checkpoint_every_optimizer_steps"]),
        evaluation_check_every_examples=int(evaluation["temperature_check_every_examples"]),
        evaluation_inter_example_sleep_seconds=float(evaluation["inter_example_sleep_seconds"]),
    )
    positive_integer_fields = (
        policy.training_check_every_micro_batches,
        policy.checkpoint_every_optimizer_steps,
        policy.evaluation_check_every_examples,
    )
    if any(value <= 0 for value in positive_integer_fields):
        raise ValueError("check and checkpoint intervals must be positive")
    if policy.checkpoint_every_optimizer_steps != 1:
        raise ValueError("formal local training must checkpoint every optimizer step")
    if policy.poll_seconds <= 0:
        raise ValueError("temperature polling interval must be positive")
    if (
        policy.training_inter_micro_batch_sleep_seconds < 0
        or policy.evaluation_inter_example_sleep_seconds < 0
    ):
        raise ValueError("pacing sleeps cannot be negative")
    return policy


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable_checkpoint(
    *,
    checkpoint_directory: Path,
    state: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Write one uniquely named checkpoint plus a hash-binding sidecar."""

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    micro_batch = int(state["next_micro_batch_index"])
    optimizer_step = int(state["optimizer_steps"])
    stem = f"checkpoint_mb_{micro_batch:04d}_step_{optimizer_step:03d}_{nonce}"
    checkpoint_path = checkpoint_directory / f"{stem}.pt"
    sidecar_path = checkpoint_directory / f"{stem}.json"
    with checkpoint_path.open("xb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    sidecar = {
        **binding,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "next_micro_batch_index": micro_batch,
        "optimizer_steps": optimizer_step,
    }
    with sidecar_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(sidecar, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "checkpoint_path": checkpoint_path,
        "sidecar_path": sidecar_path,
        "sidecar": sidecar,
    }


def load_latest_checkpoint(
    *,
    checkpoint_directory: Path,
    expected_binding: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load the most advanced valid immutable checkpoint."""

    if not checkpoint_directory.is_dir():
        return None
    valid: list[tuple[int, int, str, dict[str, Any], Path]] = []
    for sidecar_path in checkpoint_directory.glob("checkpoint_*.json"):
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if any(sidecar.get(key) != value for key, value in expected_binding.items()):
            continue
        checkpoint_name = sidecar.get("checkpoint_file")
        if not isinstance(checkpoint_name, str):
            continue
        checkpoint_path = checkpoint_directory / checkpoint_name
        if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != sidecar.get(
            "checkpoint_sha256"
        ):
            continue
        valid.append(
            (
                int(sidecar["next_micro_batch_index"]),
                int(sidecar["optimizer_steps"]),
                sidecar_path.name,
                sidecar,
                checkpoint_path,
            )
        )
    if not valid:
        return None
    _, _, _, sidecar, checkpoint_path = max(valid)
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state, dict):
        raise ValueError("checkpoint payload is not an object")
    if int(state.get("next_micro_batch_index", -1)) != int(
        sidecar["next_micro_batch_index"]
    ) or int(state.get("optimizer_steps", -1)) != int(sidecar["optimizer_steps"]):
        raise ValueError("checkpoint progress does not match its sidecar")
    return state, sidecar


def deterministic_epoch_orders(*, example_count: int, epochs: int, seed: int) -> list[list[int]]:
    """Return frozen per-epoch permutations independent of resume count."""

    if example_count <= 0 or epochs <= 0:
        raise ValueError("example_count and epochs must be positive")
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(example_count, generator=generator).tolist() for _ in range(epochs)]
