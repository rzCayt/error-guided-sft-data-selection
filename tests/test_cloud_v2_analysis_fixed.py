import copy
import json
from pathlib import Path

import pytest
import torch

from eg_sft.experiment.cloud_v2_analysis import (
    GENERATION_BATCHES,
    TRAINING_PROFILES,
    analyze_generation_calibration,
    analyze_training_calibration,
)
from eg_sft.experiment.formal_runtime import write_immutable_checkpoint


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _training_thresholds() -> dict:
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


def _make_training_run(
    root: Path,
    *,
    profile: str,
    initial_offset: float = 0.0,
    update_scale: float = 1.0,
    wall_seconds: float = 10.0,
) -> Path:
    run_dir = root / profile
    common = {
        "calibration_config_hash": "calibration",
        "protocol_config_sha256": "protocol",
        "base_recipe_config_sha256": "recipe",
        "selection_manifest_sha256": "selection",
        "selected_id_sha256": "ids",
    }
    _write_json(
        run_dir / "manifest.json",
        {
            "config": {
                **common,
                "training_profile": {"name": profile},
            }
        },
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
            "mean_response_token_loss_seen": 1.25,
            "wall_training_loop_seconds": wall_seconds,
            "compute_seconds_excluding_monitor_and_checkpoint_io": wall_seconds - 1.0,
            "peak_training_memory_gib": 8.0,
        },
    )
    initial = {
        "adapter.layer": torch.tensor(
            [initial_offset, 2.0 + initial_offset], dtype=torch.float32
        )
    }
    update = torch.tensor([1.0, -0.5], dtype=torch.float32) * update_scale
    final = {"adapter.layer": initial["adapter.layer"] + update}
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
            "optimizer_state": {
                "state": {0: {"exp_avg": torch.tensor([0.2, -0.1]) * update_scale}}
            },
            "next_micro_batch_index": 4,
            "optimizer_steps": 4,
        },
        binding=binding,
    )
    return run_dir


def _training_runs(tmp_path: Path) -> dict[str, Path]:
    return {
        profile: _make_training_run(
            tmp_path,
            profile=profile,
            wall_seconds=10.0 - index,
        )
        for index, profile in enumerate(TRAINING_PROFILES)
    }


def test_training_analysis_uses_true_update_vectors(tmp_path: Path) -> None:
    report = analyze_training_calibration(
        run_paths=_training_runs(tmp_path),
        thresholds=_training_thresholds(),
    )
    assert report["status"] == "PASS"
    by_profile = {row["profile"]: row for row in report["profiles"]}
    assert by_profile["mb4_ga4"]["update_cosine_vs_mb1"] == pytest.approx(1.0)
    assert by_profile["mb4_ga4"]["update_relative_l2_error_vs_mb1"] == 0.0
    assert "not a raw gradient" in by_profile["mb4_ga4"]["gradient_history_proxy_kind"]


def test_training_analysis_catches_inconsistent_initial_state(tmp_path: Path) -> None:
    runs = _training_runs(tmp_path)
    runs["mb2_ga8"] = _make_training_run(
        tmp_path / "changed_mb2",
        profile="mb2_ga8",
        initial_offset=3.0,
    )
    report = analyze_training_calibration(
        run_paths=runs,
        thresholds=_training_thresholds(),
    )
    row = next(item for item in report["profiles"] if item["profile"] == "mb2_ga8")
    assert row["integrity_checks"]["initial_adapter_matches_reference"] is False
    assert row["eligible"] is False


def _generation_rows() -> list[dict]:
    return [
        {
            "record_id": f"r{index}",
            "raw_output": f"Final answer: {index}",
            "parse_status": "ok",
            "parsed_prediction": str(index),
            "numeric_correct": True,
        }
        for index in range(3)
    ]


def _make_generation_run(
    root: Path,
    *,
    batch_size: int,
    rows: list[dict] | None = None,
) -> Path:
    run_dir = root / f"b{batch_size}"
    _write_json(
        run_dir / "manifest.json",
        {"config": {"generation_batch_size": batch_size}},
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "status": "PASS",
            "generation_seconds_this_invocation": 3.0 / batch_size,
            "peak_evaluation_memory_gib": 5.0 + batch_size / 10,
            "generated_token_count_this_invocation": 30,
        },
    )
    _write_jsonl(run_dir / "raw_outputs.jsonl", rows or _generation_rows())
    return run_dir


def _generation_runs(tmp_path: Path) -> dict[int, Path]:
    return {
        batch_size: _make_generation_run(tmp_path, batch_size=batch_size)
        for batch_size in GENERATION_BATCHES
    }


def test_generation_analysis_catches_record_order_error(tmp_path: Path) -> None:
    runs = _generation_runs(tmp_path)
    reordered = _generation_rows()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    runs[4] = _make_generation_run(tmp_path / "changed", batch_size=4, rows=reordered)
    report = analyze_generation_calibration(
        run_paths=runs,
        expected_count=3,
        max_difference_examples=5,
    )
    row = next(item for item in report["batches"] if item["physical_batch_size"] == 4)
    assert report["status"] == "FAIL"
    assert row["record_order_matches_batch1"] is False
    assert row["first_order_mismatch"]["index"] == 0


def test_generation_analysis_reports_prediction_difference(tmp_path: Path) -> None:
    runs = _generation_runs(tmp_path)
    changed = copy.deepcopy(_generation_rows())
    changed[1]["raw_output"] = "Final answer: 99"
    changed[1]["parsed_prediction"] = "99"
    changed[1]["numeric_correct"] = False
    runs[8] = _make_generation_run(tmp_path / "changed", batch_size=8, rows=changed)
    report = analyze_generation_calibration(
        run_paths=runs,
        expected_count=3,
        max_difference_examples=5,
    )
    row = next(item for item in report["batches"] if item["physical_batch_size"] == 8)
    assert row["prediction_equivalent_to_batch1"] is False
    assert row["field_difference_counts_vs_batch1"]["parsed_prediction"] == 1
    assert row["difference_examples"][0]["record_id"] == "r1"
    assert row["token_throughput_comparable"] is False
