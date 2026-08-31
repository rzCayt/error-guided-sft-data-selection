import copy
import json
from pathlib import Path

import pytest

import eg_sft.evaluation.cloud_v2_batching as cloud_v2_batching
from eg_sft.evaluation.cloud_v2_batching import (
    append_jsonl_rows_fsynced,
    contiguous_record_batches,
    ordered_record_ids,
)
from eg_sft.evaluation.resumable import validate_completed_prefix
from eg_sft.experiment.cloud_v2_calibration import (
    calibration_config_hash,
    calibration_run_config,
    repository_path,
    validate_calibration_config,
)
from eg_sft.experiment.formal_runtime import (
    load_latest_checkpoint,
    write_immutable_checkpoint,
)
from eg_sft.training.effective_batch import build_training_micro_batches


def _config() -> dict:
    return {
        "calibration_version": "b500-cloud-v2-calibration-v1",
        "training_example_count": 64,
        "generation_example_count": 128,
        "training_seed": 17,
        "selection_strategy": "random",
        "attention_implementation": "sdpa",
        "gradient_checkpointing": False,
        "loss_normalization": "effective_batch_response_token_sum_over_count",
        "checkpoint_every_optimizer_steps": 10,
        "generation_protocol_split": "development",
        "generation_batch_sizes": [1, 4, 8, 16],
        "training_profiles": {
            "mb1_ga16": {
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "nominal_effective_batch_size": 16,
            },
            "mb2_ga8": {
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "nominal_effective_batch_size": 16,
            },
            "mb4_ga4": {
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "nominal_effective_batch_size": 16,
            },
            "mb8_ga2": {
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "nominal_effective_batch_size": 16,
            },
        },
    }


def test_checked_in_calibration_config_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs" / "b500_cloud_v2_calibration_v1.json").read_text(
            encoding="utf-8"
        )
    )
    profiles = validate_calibration_config(payload)
    assert profiles["mb4_ga4"].nominal_effective_batch_size == 16


def test_config_hash_covers_execution_choices() -> None:
    first = _config()
    second = copy.deepcopy(first)
    second["attention_implementation"] = "eager"
    assert calibration_config_hash(first) != calibration_config_hash(second)
    profiles = validate_calibration_config(first)
    training = calibration_run_config(
        payload=first,
        profile=profiles["mb4_ga4"],
        generation_batch_size=None,
    )
    generation = calibration_run_config(
        payload=first,
        profile=None,
        generation_batch_size=8,
        adapter_sha256="a" * 64,
    )
    assert training["mode"] == "training"
    assert generation["mode"] == "generation"
    assert training["calibration_config_hash"] == generation["calibration_config_hash"]


def test_calibration_rejects_test_split_and_changed_pair() -> None:
    payload = _config()
    payload["generation_protocol_split"] = "held_out_test"
    with pytest.raises(ValueError, match="held-out"):
        validate_calibration_config(payload)
    payload = _config()
    payload["training_profiles"]["mb4_ga4"]["gradient_accumulation_steps"] = 8
    with pytest.raises(ValueError, match="frozen pair"):
        validate_calibration_config(payload)


def test_repository_paths_cannot_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        repository_path(tmp_path, "../outside", label="output")


def test_generation_batches_append_in_record_order_and_resume(tmp_path: Path) -> None:
    records = [{"record_id": f"r{index}"} for index in range(7)]
    batches = contiguous_record_batches(records=records, start_index=2, batch_size=3)
    assert [start for start, _ in batches] == [2, 5]
    assert ordered_record_ids(batches) == ["r2", "r3", "r4", "r5", "r6"]
    path = tmp_path / "raw_outputs.jsonl"
    append_jsonl_rows_fsynced(path, batches[0][1])
    append_jsonl_rows_fsynced(path, batches[1][1])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["record_id"] for row in rows] == ["r2", "r3", "r4", "r5", "r6"]
    assert validate_completed_prefix(
        completed_rows=rows,
        frozen_records=records[2:],
    ) == 5


def test_jsonl_append_fsyncs_once_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(cloud_v2_batching.os, "fsync", calls.append)
    append_jsonl_rows_fsynced(
        tmp_path / "batched.jsonl",
        [{"record_id": "a"}, {"record_id": "b"}, {"record_id": "c"}],
    )
    assert len(calls) == 1


def test_checkpoint_cursor_restores_exact_micro_batch_suffix(tmp_path: Path) -> None:
    batches = build_training_micro_batches(
        epoch_orders=[[0, 1, 2, 3, 4, 5, 6, 7]],
        micro_batch_size=2,
    )
    binding = {
        "run_config_hash": "config",
        "git_commit": "commit",
        "calibration_profile": "mb2_ga8",
        "seed": 17,
        "selected_id_sha256": "ids",
    }
    write_immutable_checkpoint(
        checkpoint_directory=tmp_path,
        state={"next_micro_batch_index": 2, "optimizer_steps": 1},
        binding=binding,
    )
    loaded = load_latest_checkpoint(
        checkpoint_directory=tmp_path,
        expected_binding=binding,
    )
    assert loaded is not None
    cursor = int(loaded[0]["next_micro_batch_index"])
    assert [item.example_index for batch in batches[cursor:] for item in batch] == [4, 5, 6, 7]
