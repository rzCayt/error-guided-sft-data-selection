import json
from pathlib import Path

import pytest
import torch

from eg_sft.experiment.formal_runtime import (
    deterministic_epoch_orders,
    load_latest_checkpoint,
    validate_execution_policy,
    write_immutable_checkpoint,
)


def _policy() -> dict:
    return {
        "execution_policy_version": "b500-formal-local-resumable-v1",
        "temperature": {
            "start_max_c": 65,
            "pause_at_c": 75,
            "resume_at_c": 62,
            "hard_stop_at_c": 80,
            "poll_seconds": 10,
        },
        "training": {
            "temperature_check_every_micro_batches": 1,
            "inter_micro_batch_sleep_seconds": 0.5,
            "checkpoint_every_optimizer_steps": 1,
        },
        "evaluation": {
            "physical_batch_size": 1,
            "temperature_check_every_examples": 1,
            "inter_example_sleep_seconds": 1,
        },
        "resources": {
            "cpu_threads": 2,
            "min_free_system_memory_gib": 8,
            "min_free_disk_gib": 100,
            "max_peak_gpu_memory_gib": 7,
        },
    }


def test_execution_policy_enforces_requested_thermal_limits() -> None:
    policy = validate_execution_policy(_policy())
    assert policy.pause_at_c == 75
    assert policy.hard_stop_at_c == 80
    assert policy.checkpoint_every_optimizer_steps == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pause_at_c", 76, "75C"),
        ("hard_stop_at_c", 81, "80C"),
        ("resume_at_c", 76, "thresholds"),
    ],
)
def test_execution_policy_rejects_weaker_temperature_guards(
    field: str,
    value: int,
    message: str,
) -> None:
    payload = _policy()
    payload["temperature"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_execution_policy(payload)


def test_epoch_orders_are_deterministic_and_complete() -> None:
    first = deterministic_epoch_orders(
        example_count=8,
        epochs=2,
        seed=17,
    )
    second = deterministic_epoch_orders(
        example_count=8,
        epochs=2,
        seed=17,
    )
    assert first == second
    assert all(sorted(epoch) == list(range(8)) for epoch in first)
    assert first[0] != first[1]


def test_latest_checkpoint_requires_matching_binding_and_hash(
    tmp_path: Path,
) -> None:
    binding = {
        "run_config_hash": "config",
        "git_commit": "commit",
        "strategy": "random",
        "seed": 17,
        "selected_id_sha256": "ids",
    }
    first = write_immutable_checkpoint(
        checkpoint_directory=tmp_path,
        state={
            "next_micro_batch_index": 16,
            "optimizer_steps": 1,
            "value": torch.tensor([1.0]),
        },
        binding=binding,
    )
    write_immutable_checkpoint(
        checkpoint_directory=tmp_path,
        state={
            "next_micro_batch_index": 32,
            "optimizer_steps": 2,
            "value": torch.tensor([2.0]),
        },
        binding=binding,
    )
    loaded = load_latest_checkpoint(
        checkpoint_directory=tmp_path,
        expected_binding=binding,
    )
    assert loaded is not None
    state, sidecar = loaded
    assert state["next_micro_batch_index"] == 32
    assert state["value"].tolist() == [2.0]
    assert sidecar["optimizer_steps"] == 2

    first_sidecar = Path(first["sidecar_path"])
    damaged = json.loads(first_sidecar.read_text(encoding="utf-8"))
    damaged["checkpoint_sha256"] = "0" * 64
    first_sidecar.write_text(
        json.dumps(damaged),
        encoding="utf-8",
    )
    loaded_after_damage = load_latest_checkpoint(
        checkpoint_directory=tmp_path,
        expected_binding=binding,
    )
    assert loaded_after_damage is not None
    assert loaded_after_damage[0]["next_micro_batch_index"] == 32
