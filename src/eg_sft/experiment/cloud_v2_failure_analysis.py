"""Analyze cloud-v2 training calibration when explicit failed profiles exist."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from eg_sft.experiment.cloud_v2_analysis import (
    COMMON_TRAINING_INPUT_FIELDS,
    TRAINING_PROFILES,
    _mapping_geometry,
    compare_update_vectors,
    load_training_run,
    optimizer_first_moment_mapping,
)
from eg_sft.training.b500 import file_sha256


FAILURE_SCHEMA_VERSION = "cloud-v2-training-calibration-failure-v1"


def validate_failure_record(
    payload: dict[str, Any],
    *,
    expected_profile: str,
) -> dict[str, Any]:
    if payload.get("failure_schema_version") != FAILURE_SCHEMA_VERSION:
        raise ValueError("unexpected calibration failure schema")
    if payload.get("profile") != expected_profile:
        raise ValueError("failure profile binding changed")
    if payload.get("status") != "FAIL":
        raise ValueError("failure artifact status must be FAIL")
    if payload.get("failure_kind") not in {
        "cuda_out_of_memory",
        "runtime_error",
        "integrity_failure",
    }:
        raise ValueError("unsupported calibration failure kind")
    if not str(payload.get("stage", "")).strip():
        raise ValueError("failure stage is required")
    exception = payload.get("exception")
    if not isinstance(exception, dict) or not str(exception.get("type", "")).strip() or not str(
        exception.get("message", "")
    ).strip():
        raise ValueError("failure exception type and message are required")
    gpu = payload.get("gpu")
    required_gpu_fields = (
        "uuid",
        "name",
        "total_memory_gib",
        "peak_allocated_memory_gib",
        "peak_reserved_memory_gib",
    )
    if not isinstance(gpu, dict) or any(field not in gpu for field in required_gpu_fields):
        raise ValueError("failure GPU identity and memory fields are required")
    input_contract = payload.get("input_contract")
    if not isinstance(input_contract, dict) or any(
        not str(input_contract.get(field, "")).strip()
        for field in COMMON_TRAINING_INPUT_FIELDS
    ):
        raise ValueError("failure input contract hashes are incomplete")
    source_log_sha256 = payload.get("source_log_sha256")
    if not isinstance(source_log_sha256, str) or len(source_log_sha256) != 64:
        raise ValueError("failure artifact requires a source log SHA-256")
    if not str(payload.get("recorded_at_utc", "")).strip():
        raise ValueError("failure recorded_at_utc is required")
    return payload


def _successful_profile_row(
    *,
    profile: str,
    run: dict[str, Any],
    reference: dict[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_config = run["manifest"]["config"]
    reference_config = reference["manifest"]["config"]
    metrics = run["metrics"]
    reference_metrics = reference["metrics"]
    reference_tokens = int(reference_metrics["supervised_tokens_seen"])
    reference_loss = float(reference_metrics["mean_response_token_loss_seen"])
    integrity_checks = {
        "status_pass": metrics.get("status") == "PASS",
        "input_hashes_match_reference": all(
            manifest_config.get(field) == reference_config.get(field)
            for field in COMMON_TRAINING_INPUT_FIELDS
        ),
        "initial_adapter_matches_reference": (
            run["initial_adapter_sha256"] == reference["initial_adapter_sha256"]
        ),
        "optimizer_steps_match": (
            int(metrics.get("optimizer_steps_planned", -1))
            == int(thresholds["expected_optimizer_steps"])
            and int(metrics.get("optimizer_steps_completed", -1))
            == int(thresholds["expected_optimizer_steps"])
        ),
        "supervised_tokens_match_reference": (
            int(metrics.get("supervised_tokens_seen", -1)) == reference_tokens
        ),
        "temperature_sampling_matches": (
            int(metrics.get("temperature_sample_count", -1))
            == int(thresholds["expected_temperature_sample_count"])
            and metrics.get("temperature_sampling_rule")
            == "once_at_start_and_once_per_optimizer_boundary"
        ),
        "adapter_reload_pass": (
            float(metrics.get("adapter_reload_loss_absolute_difference", math.inf))
            <= float(thresholds["adapter_reload_loss_difference_max"])
        ),
    }
    update = compare_update_vectors(
        reference_initial=reference["initial_adapter"],
        reference_final=reference["final_adapter"],
        candidate_initial=run["initial_adapter"],
        candidate_final=run["final_adapter"],
    )
    gradient_proxy = _mapping_geometry(
        optimizer_first_moment_mapping(reference["final"]),
        optimizer_first_moment_mapping(run["final"]),
    )
    loss = float(metrics["mean_response_token_loss_seen"])
    loss_relative_difference = abs(loss - reference_loss) / max(abs(reference_loss), 1e-12)
    numerical_checks = {
        "update_cosine": update["cosine"] >= float(thresholds["update_cosine_min"]),
        "update_relative_l2": update["relative_l2_error"]
        <= float(thresholds["update_relative_l2_error_max"]),
        "gradient_history_proxy_cosine": gradient_proxy["cosine"]
        >= float(thresholds["gradient_history_proxy_cosine_min"]),
        "loss_relative_difference": loss_relative_difference
        <= float(thresholds["loss_relative_difference_max"]),
    }
    wall_seconds = float(metrics["wall_training_loop_seconds"])
    return {
        "profile": profile,
        "outcome": "PASS_RUN",
        "integrity_checks": integrity_checks,
        "integrity_pass": all(integrity_checks.values()),
        "numerical_checks": numerical_checks,
        "numerical_pass": all(numerical_checks.values()),
        "eligible": all(integrity_checks.values()) and all(numerical_checks.values()),
        "update_cosine_vs_mb1": update["cosine"],
        "update_relative_l2_error_vs_mb1": update["relative_l2_error"],
        "gradient_history_proxy_cosine_vs_mb1": gradient_proxy["cosine"],
        "gradient_history_proxy_kind": "final AdamW exp_avg; not a raw gradient",
        "loss": loss,
        "loss_relative_difference_vs_mb1": loss_relative_difference,
        "wall_training_loop_seconds": wall_seconds,
        "wall_tokens_per_second": reference_tokens / wall_seconds,
        "peak_allocated_memory_gib": metrics.get("peak_training_memory_gib"),
        "peak_reserved_memory_gib": metrics.get("peak_training_reserved_memory_gib"),
    }


def analyze_training_calibration_with_failures(
    *,
    run_paths: Mapping[str, Path],
    failure_paths: Mapping[str, Path],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if set(run_paths) & set(failure_paths):
        raise ValueError("a profile cannot be both a run and a failure")
    if set(run_paths) | set(failure_paths) != set(TRAINING_PROFILES):
        raise ValueError("runs and failures must cover all four frozen profiles")
    if "mb1_ga16" not in run_paths:
        raise ValueError("mb1_ga16 reference must be a successful run")
    runs = {
        profile: load_training_run(path, expected_profile=profile)
        for profile, path in run_paths.items()
    }
    reference = runs["mb1_ga16"]
    reference_contract = reference["manifest"]["config"]
    failures: dict[str, dict[str, Any]] = {}
    for profile, path in failure_paths.items():
        from eg_sft.experiment.cloud_v2_analysis import read_json_object

        payload = validate_failure_record(
            read_json_object(path),
            expected_profile=profile,
        )
        failures[profile] = payload

    rows: list[dict[str, Any]] = []
    successful_rows: dict[str, dict[str, Any]] = {}
    for profile in TRAINING_PROFILES:
        if profile in runs:
            row = _successful_profile_row(
                profile=profile,
                run=runs[profile],
                reference=reference,
                thresholds=thresholds,
            )
            successful_rows[profile] = row
            rows.append(row)
            continue
        failure = failures[profile]
        input_matches = all(
            failure["input_contract"].get(field) == reference_contract.get(field)
            for field in COMMON_TRAINING_INPUT_FIELDS
        )
        rows.append(
            {
                "profile": profile,
                "outcome": "FAILURE_ARTIFACT",
                "eligible": False,
                "elimination_reason": failure["failure_kind"],
                "stage": failure["stage"],
                "exception": failure["exception"],
                "gpu": failure["gpu"],
                "input_contract_matches_reference": input_matches,
                "source_log_sha256": failure["source_log_sha256"],
                "failure_artifact_sha256": file_sha256(failure_paths[profile]),
                "recorded_at_utc": failure["recorded_at_utc"],
            }
        )

    selected_profile: str | None = None
    if successful_rows.get("mb4_ga4", {}).get("eligible"):
        selected_profile = "mb4_ga4"
        mb8 = successful_rows.get("mb8_ga2")
        if mb8 and mb8["eligible"] and mb8["wall_tokens_per_second"] >= (
            successful_rows["mb4_ga4"]["wall_tokens_per_second"]
            * (1.0 + float(thresholds["mb8_speed_advantage_required"]))
        ):
            selected_profile = "mb8_ga2"
    elif successful_rows.get("mb2_ga8", {}).get("eligible"):
        selected_profile = "mb2_ga8"
    elif successful_rows["mb1_ga16"]["eligible"]:
        selected_profile = "mb1_ga16"
    return {
        "analysis_version": "cloud-v2-training-calibration-with-failures-v1",
        "status": (
            "PASS_WITH_REJECTED_PROFILES" if selected_profile is not None else "FAIL"
        ),
        "reference_profile": "mb1_ga16",
        "profiles": rows,
        "selected_profile": selected_profile,
        "failure_count": len(failures),
        "formal_matrix_action": (
            "freeze only the selected passing profile"
            if selected_profile is not None
            else "do not start cloud-v2 formal matrix"
        ),
        "evidence_boundary": (
            "Failure profiles are eliminated without fabricated vectors. Raw gradients were "
            "not saved; AdamW exp_avg remains only a gradient-history proxy."
        ),
    }
