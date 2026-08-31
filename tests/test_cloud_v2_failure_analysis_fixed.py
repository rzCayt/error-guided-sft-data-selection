import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from eg_sft.experiment.cloud_v2_failure_analysis import (
    analyze_training_calibration_with_failures,
    validate_failure_record,
)
from eg_sft.experiment.formal_runtime import write_immutable_checkpoint


COMMON = {
    "calibration_config_hash": "calibration",
    "protocol_config_sha256": "protocol",
    "base_recipe_config_sha256": "recipe",
    "selection_manifest_sha256": "selection",
    "selected_id_sha256": "ids",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_run(root: Path, profile: str, wall_seconds: float) -> Path:
    run_dir = root / profile
    _write_json(
        run_dir / "manifest.json",
        {"config": {**COMMON, "training_profile": {"name": profile}}},
    )
    _write_json(
        run_dir / "training_complete" / "calibration_metrics.json",
        {
            "status": "PASS",
            "profile": profile,
            "optimizer_steps_planned": 4,
            "optimizer_steps_completed": 4,
            "supervised_tokens_seen": 128,
            "temperature_sample_count": 5,
            "temperature_sampling_rule": "once_at_start_and_once_per_optimizer_boundary",
            "adapter_reload_loss_absolute_difference": 0.0,
            "mean_response_token_loss_seen": 1.0,
            "wall_training_loop_seconds": wall_seconds,
            "peak_training_memory_gib": 8.0,
        },
    )
    initial = {"adapter": torch.tensor([0.0, 2.0])}
    final = {"adapter": torch.tensor([1.0, 1.5])}
    binding = {"profile": profile}
    write_immutable_checkpoint(
        checkpoint_directory=run_dir / "checkpoints",
        state={
            "adapter_state": initial,
            "optimizer_state": {"state": {}},
            "next_micro_batch_index": 0,
            "optimizer_steps": 0,
        },
        binding=binding,
    )
    write_immutable_checkpoint(
        checkpoint_directory=run_dir / "checkpoints",
        state={
            "adapter_state": final,
            "optimizer_state": {"state": {0: {"exp_avg": torch.tensor([0.2, -0.1])}}},
            "next_micro_batch_index": 4,
            "optimizer_steps": 4,
        },
        binding=binding,
    )
    return run_dir


def _failure() -> dict:
    return {
        "failure_schema_version": "cloud-v2-training-calibration-failure-v1",
        "profile": "mb8_ga2",
        "status": "FAIL",
        "failure_kind": "cuda_out_of_memory",
        "stage": "forward_backward_micro_batch_1",
        "exception": {
            "type": "torch.OutOfMemoryError",
            "message": "synthetic test OOM",
        },
        "gpu": {
            "uuid": "GPU-test",
            "name": "RTX-test",
            "total_memory_gib": 24.0,
            "peak_allocated_memory_gib": 20.5,
            "peak_reserved_memory_gib": 23.0,
        },
        "input_contract": dict(COMMON),
        "source_log_sha256": "f" * 64,
        "recorded_at_utc": "2026-08-23T12:00:00+00:00",
    }


def _thresholds() -> dict:
    return {
        "expected_optimizer_steps": 4,
        "expected_temperature_sample_count": 5,
        "adapter_reload_loss_difference_max": 1e-6,
        "update_cosine_min": 0.999,
        "update_relative_l2_error_max": 0.01,
        "gradient_history_proxy_cosine_min": 0.999,
        "loss_relative_difference_max": 0.01,
        "mb8_speed_advantage_required": 0.15,
    }


def test_one_oom_and_three_pass_runs_selects_mb4_without_fabricating_vector(
    tmp_path: Path,
) -> None:
    runs = {
        "mb1_ga16": _make_run(tmp_path, "mb1_ga16", 10.0),
        "mb2_ga8": _make_run(tmp_path, "mb2_ga8", 8.0),
        "mb4_ga4": _make_run(tmp_path, "mb4_ga4", 6.0),
    }
    failure_path = tmp_path / "mb8_oom.json"
    _write_json(failure_path, _failure())
    report = analyze_training_calibration_with_failures(
        run_paths=runs,
        failure_paths={"mb8_ga2": failure_path},
        thresholds=_thresholds(),
    )
    assert report["status"] == "PASS_WITH_REJECTED_PROFILES"
    assert report["selected_profile"] == "mb4_ga4"
    mb8 = next(row for row in report["profiles"] if row["profile"] == "mb8_ga2")
    assert mb8["eligible"] is False
    assert mb8["elimination_reason"] == "cuda_out_of_memory"
    assert "update_cosine_vs_mb1" not in mb8
    assert mb8["gpu"]["peak_reserved_memory_gib"] == 23.0


def test_failure_schema_rejects_missing_memory_evidence() -> None:
    payload = _failure()
    del payload["gpu"]["peak_reserved_memory_gib"]
    with pytest.raises(ValueError, match="memory fields"):
        validate_failure_record(payload, expected_profile="mb8_ga2")


def test_failure_aware_cli_help_is_cpu_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_cloud_v2_training_calibration_with_failures.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "--failure" in process.stdout
