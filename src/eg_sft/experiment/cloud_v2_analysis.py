"""Offline integrity and equivalence analysis for cloud-v2 calibration runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from eg_sft.training.b500 import file_sha256, read_jsonl


TRAINING_PROFILES = ("mb1_ga16", "mb2_ga8", "mb4_ga4", "mb8_ga2")
GENERATION_BATCHES = (1, 4, 8, 16)
COMMON_TRAINING_INPUT_FIELDS = (
    "calibration_config_hash",
    "protocol_config_sha256",
    "base_recipe_config_sha256",
    "selection_manifest_sha256",
    "selected_id_sha256",
)


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def parse_named_paths(
    values: Sequence[str],
    *,
    expected_names: Sequence[str],
) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"expected NAME=PATH, received {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate run name: {name}")
        parsed[name] = Path(raw_path).resolve()
    expected = set(expected_names)
    if set(parsed) != expected:
        raise ValueError(
            f"run names must be exactly {sorted(expected)}; received {sorted(parsed)}"
        )
    return parsed


def tensor_mapping_sha256(mapping: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        tensor = mapping[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _mapping_geometry(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    if set(reference) != set(candidate):
        raise ValueError("tensor mapping keys differ")
    dot = 0.0
    reference_norm_sq = 0.0
    candidate_norm_sq = 0.0
    difference_norm_sq = 0.0
    for name in sorted(reference):
        left = reference[name].detach().cpu()
        right = candidate[name].detach().cpu()
        if left.shape != right.shape:
            raise ValueError(f"tensor shape differs for {name}")
        left64 = left.to(torch.float64).reshape(-1)
        right64 = right.to(torch.float64).reshape(-1)
        difference = right64 - left64
        dot += float(torch.dot(left64, right64).item())
        reference_norm_sq += float(torch.dot(left64, left64).item())
        candidate_norm_sq += float(torch.dot(right64, right64).item())
        difference_norm_sq += float(torch.dot(difference, difference).item())
    reference_norm = math.sqrt(reference_norm_sq)
    candidate_norm = math.sqrt(candidate_norm_sq)
    if reference_norm == 0.0 or candidate_norm == 0.0:
        raise ValueError("cannot compare a zero vector")
    return {
        "cosine": dot / (reference_norm * candidate_norm),
        "reference_l2_norm": reference_norm,
        "candidate_l2_norm": candidate_norm,
        "difference_l2_norm": math.sqrt(difference_norm_sq),
        "relative_l2_error": math.sqrt(difference_norm_sq) / reference_norm,
    }


def compare_update_vectors(
    *,
    reference_initial: Mapping[str, torch.Tensor],
    reference_final: Mapping[str, torch.Tensor],
    candidate_initial: Mapping[str, torch.Tensor],
    candidate_final: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Compare true adapter updates, never the final parameters themselves."""

    keys = set(reference_initial)
    if not (
        keys
        == set(reference_final)
        == set(candidate_initial)
        == set(candidate_final)
    ):
        raise ValueError("adapter checkpoint keys differ")
    reference_update: dict[str, torch.Tensor] = {}
    candidate_update: dict[str, torch.Tensor] = {}
    for name in sorted(keys):
        tensors = (
            reference_initial[name],
            reference_final[name],
            candidate_initial[name],
            candidate_final[name],
        )
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            raise ValueError(f"adapter checkpoint shape differs for {name}")
        reference_update[name] = reference_final[name].detach().cpu() - reference_initial[
            name
        ].detach().cpu()
        candidate_update[name] = candidate_final[name].detach().cpu() - candidate_initial[
            name
        ].detach().cpu()
    return _mapping_geometry(reference_update, candidate_update)


def optimizer_first_moment_mapping(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return AdamW exp_avg as a labelled gradient-history proxy."""

    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(optimizer_state, dict) or not isinstance(
        optimizer_state.get("state"), dict
    ):
        raise ValueError("checkpoint has no optimizer state")
    mapping: dict[str, torch.Tensor] = {}
    for parameter_id, state in sorted(
        optimizer_state["state"].items(), key=lambda item: int(item[0])
    ):
        if not isinstance(state, dict) or not isinstance(state.get("exp_avg"), torch.Tensor):
            raise ValueError("optimizer state has no AdamW exp_avg tensor")
        mapping[str(parameter_id)] = state["exp_avg"].detach().cpu()
    if not mapping:
        raise ValueError("optimizer first-moment mapping is empty")
    return mapping


def _load_valid_checkpoints(run_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    checkpoint_dir = run_dir / "checkpoints"
    valid: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for sidecar_path in sorted(checkpoint_dir.glob("checkpoint_*.json")):
        sidecar = read_json_object(sidecar_path)
        checkpoint_name = sidecar.get("checkpoint_file")
        if not isinstance(checkpoint_name, str):
            continue
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.is_file():
            continue
        if file_sha256(checkpoint_path) != sidecar.get("checkpoint_sha256"):
            continue
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint is not an object: {checkpoint_path}")
        if int(state.get("optimizer_steps", -1)) != int(sidecar["optimizer_steps"]):
            raise ValueError("checkpoint optimizer step differs from sidecar")
        if int(state.get("next_micro_batch_index", -1)) != int(
            sidecar["next_micro_batch_index"]
        ):
            raise ValueError("checkpoint cursor differs from sidecar")
        valid.append((state, sidecar, checkpoint_path))
    if not valid:
        raise ValueError(f"no valid checkpoints in {checkpoint_dir}")
    return valid


def load_training_run(run_dir: Path, *, expected_profile: str) -> dict[str, Any]:
    manifest = read_json_object(run_dir / "manifest.json")
    metrics = read_json_object(run_dir / "training_complete" / "calibration_metrics.json")
    manifest_profile = manifest.get("config", {}).get("training_profile", {}).get("name")
    if manifest_profile != expected_profile or metrics.get("profile") != expected_profile:
        raise ValueError(f"profile binding mismatch for {expected_profile}")
    checkpoints = _load_valid_checkpoints(run_dir)
    initial_candidates = [row for row in checkpoints if int(row[0]["optimizer_steps"]) == 0]
    planned_steps = int(metrics["optimizer_steps_planned"])
    final_candidates = [
        row for row in checkpoints if int(row[0]["optimizer_steps"]) == planned_steps
    ]
    if len(initial_candidates) != 1:
        raise ValueError(f"{expected_profile} must have exactly one initial checkpoint")
    if len(final_candidates) != 1:
        raise ValueError(f"{expected_profile} must have exactly one final checkpoint")
    initial, initial_sidecar, initial_path = initial_candidates[0]
    final, final_sidecar, final_path = final_candidates[0]
    initial_adapter = initial.get("adapter_state")
    final_adapter = final.get("adapter_state")
    if not isinstance(initial_adapter, dict) or not isinstance(final_adapter, dict):
        raise ValueError("checkpoint adapter_state is missing")
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "metrics": metrics,
        "initial": initial,
        "final": final,
        "initial_adapter": initial_adapter,
        "final_adapter": final_adapter,
        "initial_adapter_sha256": tensor_mapping_sha256(initial_adapter),
        "initial_checkpoint_file": initial_path.name,
        "initial_checkpoint_sha256": initial_sidecar["checkpoint_sha256"],
        "final_checkpoint_file": final_path.name,
        "final_checkpoint_sha256": final_sidecar["checkpoint_sha256"],
    }


def analyze_training_calibration(
    *,
    run_paths: Mapping[str, Path],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if set(run_paths) != set(TRAINING_PROFILES):
        raise ValueError("training analysis requires exactly four frozen profiles")
    runs = {
        profile: load_training_run(path, expected_profile=profile)
        for profile, path in run_paths.items()
    }
    reference = runs["mb1_ga16"]
    reference_manifest = reference["manifest"]["config"]
    reference_metrics = reference["metrics"]
    reference_initial_sha = reference["initial_adapter_sha256"]
    reference_tokens = int(reference_metrics["supervised_tokens_seen"])
    reference_loss = float(reference_metrics["mean_response_token_loss_seen"])
    expected_steps = int(thresholds["expected_optimizer_steps"])
    expected_temperature_samples = int(thresholds["expected_temperature_sample_count"])
    max_reload_difference = float(thresholds["adapter_reload_loss_difference_max"])

    rows: list[dict[str, Any]] = []
    comparison_by_profile: dict[str, dict[str, Any]] = {}
    for profile in TRAINING_PROFILES:
        run = runs[profile]
        manifest_config = run["manifest"]["config"]
        metrics = run["metrics"]
        input_matches = all(
            manifest_config.get(field) == reference_manifest.get(field)
            for field in COMMON_TRAINING_INPUT_FIELDS
        )
        integrity_checks = {
            "status_pass": metrics.get("status") == "PASS",
            "input_hashes_match_reference": input_matches,
            "initial_adapter_matches_reference": (
                run["initial_adapter_sha256"] == reference_initial_sha
            ),
            "optimizer_steps_match": (
                int(metrics.get("optimizer_steps_planned", -1)) == expected_steps
                and int(metrics.get("optimizer_steps_completed", -1)) == expected_steps
            ),
            "supervised_tokens_match_reference": (
                int(metrics.get("supervised_tokens_seen", -1)) == reference_tokens
            ),
            "temperature_sampling_matches": (
                int(metrics.get("temperature_sample_count", -1))
                == expected_temperature_samples
                and metrics.get("temperature_sampling_rule")
                == "once_at_start_and_once_per_optimizer_boundary"
            ),
            "adapter_reload_pass": (
                float(metrics.get("adapter_reload_loss_absolute_difference", math.inf))
                <= max_reload_difference
            ),
        }
        update_geometry = compare_update_vectors(
            reference_initial=reference["initial_adapter"],
            reference_final=reference["final_adapter"],
            candidate_initial=run["initial_adapter"],
            candidate_final=run["final_adapter"],
        )
        gradient_proxy_geometry = _mapping_geometry(
            optimizer_first_moment_mapping(reference["final"]),
            optimizer_first_moment_mapping(run["final"]),
        )
        candidate_loss = float(metrics["mean_response_token_loss_seen"])
        loss_absolute_difference = abs(candidate_loss - reference_loss)
        loss_relative_difference = loss_absolute_difference / max(abs(reference_loss), 1e-12)
        numeric_checks = {
            "update_cosine": (
                update_geometry["cosine"] >= float(thresholds["update_cosine_min"])
            ),
            "update_relative_l2": (
                update_geometry["relative_l2_error"]
                <= float(thresholds["update_relative_l2_error_max"])
            ),
            "gradient_history_proxy_cosine": (
                gradient_proxy_geometry["cosine"]
                >= float(thresholds["gradient_history_proxy_cosine_min"])
            ),
            "loss_relative_difference": (
                loss_relative_difference
                <= float(thresholds["loss_relative_difference_max"])
            ),
        }
        integrity_pass = all(integrity_checks.values())
        numerical_pass = all(numeric_checks.values())
        allocated = metrics.get("peak_training_memory_gib")
        reserved = metrics.get("peak_training_reserved_memory_gib")
        wall_seconds = float(metrics["wall_training_loop_seconds"])
        compute_seconds = float(
            metrics["compute_seconds_excluding_monitor_and_checkpoint_io"]
        )
        row = {
            "profile": profile,
            "run_dir": str(run["run_dir"]),
            "integrity_checks": integrity_checks,
            "integrity_pass": integrity_pass,
            "numerical_checks": numeric_checks,
            "numerical_pass": numerical_pass,
            "eligible": integrity_pass and numerical_pass,
            "initial_adapter_sha256": run["initial_adapter_sha256"],
            "initial_checkpoint_sha256": run["initial_checkpoint_sha256"],
            "final_checkpoint_sha256": run["final_checkpoint_sha256"],
            "update_cosine_vs_mb1": update_geometry["cosine"],
            "update_relative_l2_error_vs_mb1": update_geometry["relative_l2_error"],
            "update_l2_norm": update_geometry["candidate_l2_norm"],
            "gradient_history_proxy_kind": "final AdamW exp_avg; not a raw gradient",
            "gradient_history_proxy_cosine_vs_mb1": gradient_proxy_geometry["cosine"],
            "loss": candidate_loss,
            "loss_absolute_difference_vs_mb1": loss_absolute_difference,
            "loss_relative_difference_vs_mb1": loss_relative_difference,
            "supervised_tokens_seen": int(metrics["supervised_tokens_seen"]),
            "wall_training_loop_seconds": wall_seconds,
            "compute_seconds": compute_seconds,
            "wall_tokens_per_second": reference_tokens / wall_seconds,
            "compute_tokens_per_second": reference_tokens / compute_seconds,
            "peak_allocated_memory_gib": allocated,
            "peak_reserved_memory_gib": reserved,
            "reserved_memory_recorded": reserved is not None,
        }
        rows.append(row)
        comparison_by_profile[profile] = row

    eligible = [row for row in rows if row["eligible"]]
    selected_profile: str | None = None
    if comparison_by_profile["mb4_ga4"]["eligible"]:
        selected_profile = "mb4_ga4"
        mb8 = comparison_by_profile["mb8_ga2"]
        if mb8["eligible"] and mb8["wall_tokens_per_second"] >= (
            comparison_by_profile["mb4_ga4"]["wall_tokens_per_second"]
            * (1.0 + float(thresholds["mb8_speed_advantage_required"]))
        ):
            selected_profile = "mb8_ga2"
    elif eligible:
        selected_profile = max(eligible, key=lambda row: row["wall_tokens_per_second"])[
            "profile"
        ]
    return {
        "analysis_version": "cloud-v2-training-calibration-analysis-v1",
        "status": "PASS" if selected_profile is not None else "FAIL",
        "reference_profile": "mb1_ga16",
        "gradient_evidence_boundary": (
            "Raw gradients were not checkpointed. AdamW exp_avg is reported only as a "
            "gradient-history proxy; adapter cosine always uses final-minus-initial updates."
        ),
        "thresholds": dict(thresholds),
        "profiles": rows,
        "selected_profile": selected_profile,
        "formal_matrix_action": (
            "freeze a new formal cloud-v2 contract"
            if selected_profile is not None
            else "do not start cloud-v2 formal matrix"
        ),
    }


def load_generation_run(
    run_dir: Path,
    *,
    expected_batch_size: int,
) -> dict[str, Any]:
    manifest = read_json_object(run_dir / "manifest.json")
    metrics = read_json_object(run_dir / "metrics.json")
    observed_batch = int(manifest.get("config", {}).get("generation_batch_size", -1))
    if observed_batch != expected_batch_size:
        raise ValueError(f"generation batch binding mismatch for {expected_batch_size}")
    rows = read_jsonl(run_dir / "raw_outputs.jsonl")
    return {"run_dir": run_dir, "manifest": manifest, "metrics": metrics, "rows": rows}


def _first_order_mismatch(reference_ids: list[str], candidate_ids: list[str]) -> dict[str, Any] | None:
    for index, (reference, candidate) in enumerate(
        zip(reference_ids, candidate_ids, strict=False)
    ):
        if reference != candidate:
            return {"index": index, "reference_record_id": reference, "candidate_record_id": candidate}
    if len(reference_ids) != len(candidate_ids):
        return {
            "index": min(len(reference_ids), len(candidate_ids)),
            "reference_count": len(reference_ids),
            "candidate_count": len(candidate_ids),
        }
    return None


def analyze_generation_calibration(
    *,
    run_paths: Mapping[int, Path],
    expected_count: int,
    max_difference_examples: int,
) -> dict[str, Any]:
    if set(run_paths) != set(GENERATION_BATCHES):
        raise ValueError("generation analysis requires batch sizes 1, 4, 8, and 16")
    runs = {
        batch_size: load_generation_run(path, expected_batch_size=batch_size)
        for batch_size, path in run_paths.items()
    }
    reference = runs[1]
    reference_rows = reference["rows"]
    reference_ids = [str(row.get("record_id", "")) for row in reference_rows]
    reference_by_id = {str(row["record_id"]): row for row in reference_rows}
    if len(reference_rows) != expected_count or len(reference_by_id) != expected_count:
        raise ValueError("batch-1 reference count or record IDs are invalid")
    comparison_fields = ("raw_output", "parse_status", "parsed_prediction", "numeric_correct")
    results: list[dict[str, Any]] = []
    for batch_size in GENERATION_BATCHES:
        run = runs[batch_size]
        rows = run["rows"]
        ids = [str(row.get("record_id", "")) for row in rows]
        order_mismatch = _first_order_mismatch(reference_ids, ids)
        unique_ids = len(set(ids)) == len(ids)
        same_id_set = set(ids) == set(reference_ids)
        candidate_by_id = {str(row["record_id"]): row for row in rows} if unique_ids else {}
        field_difference_counts = {field: 0 for field in comparison_fields}
        difference_examples: list[dict[str, Any]] = []
        if unique_ids and same_id_set:
            for record_id in reference_ids:
                left = reference_by_id[record_id]
                right = candidate_by_id[record_id]
                changed = [field for field in comparison_fields if left.get(field) != right.get(field)]
                for field in changed:
                    field_difference_counts[field] += 1
                if changed and len(difference_examples) < max_difference_examples:
                    difference_examples.append(
                        {
                            "record_id": record_id,
                            "changed_fields": changed,
                            "reference": {field: left.get(field) for field in changed},
                            "candidate": {field: right.get(field) for field in changed},
                        }
                    )
        metrics = run["metrics"]
        wall_seconds = float(metrics["generation_seconds_this_invocation"])
        unpadded_tokens = metrics.get("unpadded_generated_token_count_this_invocation")
        token_throughput_comparable = unpadded_tokens is not None
        result = {
            "physical_batch_size": batch_size,
            "run_dir": str(run["run_dir"]),
            "example_count": len(rows),
            "record_count_matches": len(rows) == expected_count,
            "record_ids_unique": unique_ids,
            "record_id_set_matches_batch1": same_id_set,
            "record_order_matches_batch1": order_mismatch is None,
            "first_order_mismatch": order_mismatch,
            "field_difference_counts_vs_batch1": field_difference_counts,
            "difference_examples": difference_examples,
            "wall_seconds": wall_seconds,
            "wall_examples_per_second": len(rows) / wall_seconds,
            "peak_allocated_memory_gib": metrics.get("peak_evaluation_memory_gib"),
            "peak_reserved_memory_gib": metrics.get("peak_evaluation_reserved_memory_gib"),
            "reported_generated_token_count": metrics.get(
                "generated_token_count_this_invocation"
            ),
            "unpadded_generated_token_count": unpadded_tokens,
            "token_throughput_comparable": token_throughput_comparable,
            "token_throughput_note": (
                "Comparable unpadded token count was recorded."
                if token_throughput_comparable
                else "Existing count includes batch padding after EOS; do not compare token/s."
            ),
        }
        result["integrity_pass"] = all(
            (
                result["record_count_matches"],
                result["record_ids_unique"],
                result["record_id_set_matches_batch1"],
                result["record_order_matches_batch1"],
            )
        )
        result["prediction_equivalent_to_batch1"] = all(
            count == 0 for count in field_difference_counts.values()
        )
        results.append(result)
    overall_pass = all(
        row["integrity_pass"] and row["prediction_equivalent_to_batch1"] for row in results
    )
    return {
        "analysis_version": "cloud-v2-generation-calibration-analysis-v1",
        "status": "PASS" if overall_pass else "FAIL",
        "reference_batch_size": 1,
        "primary_throughput_metric": "wall_examples_per_second",
        "generated_token_boundary": (
            "Token/s is excluded unless an explicitly unpadded generated-token count exists."
        ),
        "batches": results,
    }
