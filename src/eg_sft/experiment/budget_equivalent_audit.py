"""Read-only integrity audit for one budget-equivalent Phase 1 cell."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from eg_sft.experiment.b500_engineering_audit import audit_completed_evaluation
from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract
from eg_sft.training.b500 import file_sha256, read_jsonl
from eg_sft.training.token_budget import supervision_tokens_per_step


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"invalid SHA-256 field: {field}")
    return text


def audit_dose_only_token_cap_artifacts(
    *,
    training_metrics: dict[str, Any],
    token_budget_audit: dict[str, Any],
    optimizer_step_rows: list[dict[str, Any]],
    supervision_token_cap: int,
    token_cap_policy: str,
) -> dict[str, Any]:
    """Fail closed on every mask and coverage field in a dose-only run."""

    expected_steps = 64
    expected_per_step = supervision_tokens_per_step(
        supervision_token_cap=supervision_token_cap,
        optimizer_steps=expected_steps,
        policy=token_cap_policy,
    )
    if len(optimizer_step_rows) != expected_steps:
        raise ValueError("dose-only audit requires exactly 64 optimizer-step rows")
    selected_token_set_sha256 = _require_sha256(
        optimizer_step_rows[0].get("selected_token_set_sha256"),
        field="selected_token_set_sha256",
    )
    boundary_split_occurrence_count = int(
        optimizer_step_rows[0].get("boundary_split_occurrence_count", -1)
    )
    mask_shas: list[str] = []
    for index, row in enumerate(optimizer_step_rows, start=1):
        mask_sha = _require_sha256(
            row.get("token_cap_mask_sha256"),
            field=f"optimizer_step[{index}].token_cap_mask_sha256",
        )
        mask_shas.append(mask_sha)
        if (
            int(row.get("optimizer_step", -1)) != index
            or int(row.get("response_supervision_tokens", -1)) != expected_per_step
            or int(row.get("kept_response_supervision_tokens", -1))
            != expected_per_step
            or int(row.get("candidate_response_supervision_tokens", -1))
            < expected_per_step
            or int(row.get("cumulative_response_supervision_tokens", -1))
            != index * expected_per_step
            or row.get("token_cap_policy") != token_cap_policy
            or row.get("selected_token_set_sha256") != selected_token_set_sha256
            or row.get("legacy_sequence_step_boundaries_preserved") is not False
            or int(row.get("boundary_split_occurrence_count", -1))
            != boundary_split_occurrence_count
            or int(row.get("selected_candidate_id_coverage", -1)) != 500
            or int(row.get("candidate_id_count", -1)) != 500
            or int(row.get("occurrence_with_kept_token_count", -1)) != 1000
            or int(row.get("occurrence_count", -1)) != 1000
            or int(row.get("mandatory_coverage_token_count", -1)) != 1000
        ):
            raise ValueError(f"dose-only optimizer-step evidence changed at step {index}")
    if boundary_split_occurrence_count < 0:
        raise ValueError("dose-only boundary split count is invalid")
    mask_set_text = "\n".join(mask_shas) + "\n"
    mask_set_sha256 = hashlib.sha256(mask_set_text.encode("utf-8")).hexdigest()
    expected_fields = {
        "supervision_token_cap": supervision_token_cap,
        "supervision_tokens_per_optimizer_step": expected_per_step,
        "token_cap_policy": token_cap_policy,
        "selected_token_set_sha256": selected_token_set_sha256,
        "legacy_sequence_step_boundaries_preserved": False,
        "boundary_split_occurrence_count": boundary_split_occurrence_count,
        "selected_candidate_id_coverage": 500,
        "candidate_id_count": 500,
        "occurrence_with_kept_token_count": 1000,
        "occurrence_count": 1000,
        "mandatory_coverage_token_count": 1000,
    }
    for artifact_name, artifact in (
        ("training_metrics", training_metrics),
        ("token_budget_audit", token_budget_audit),
    ):
        for field, value in expected_fields.items():
            if artifact.get(field) != value:
                raise ValueError(f"{artifact_name} dose-only field changed: {field}")
    if int(training_metrics.get("supervised_tokens_seen", -1)) != supervision_token_cap:
        raise ValueError("training metrics dose-only exposure changed")
    if (
        int(token_budget_audit.get("response_supervision_exposure_tokens", -1))
        != supervision_token_cap
        or token_budget_audit.get("optimizer_step_mask_set_sha256")
        != mask_set_sha256
    ):
        raise ValueError("token budget dose-only aggregate evidence changed")
    return {
        "status": "PASS",
        "optimizer_steps": expected_steps,
        "tokens_per_optimizer_step": expected_per_step,
        "supervision_token_cap": supervision_token_cap,
        "token_cap_policy": token_cap_policy,
        "selected_token_set_sha256": selected_token_set_sha256,
        "optimizer_step_mask_set_sha256": mask_set_sha256,
        "selected_candidate_id_coverage": 500,
        "occurrence_with_kept_token_count": 1000,
        "boundary_split_occurrence_count": boundary_split_occurrence_count,
    }


def audit_training_artifacts(
    *,
    training_metrics: dict[str, Any],
    token_budget_audit: dict[str, Any],
    token_audit: list[dict[str, Any]],
    optimizer_step_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if training_metrics.get("status") != "PASS":
        raise ValueError("training status is not PASS")
    if int(training_metrics.get("selected_count", -1)) != 500:
        raise ValueError("training selected count changed")
    if int(training_metrics.get("optimizer_steps_planned", -1)) != 64:
        raise ValueError("planned optimizer steps changed")
    if int(training_metrics.get("optimizer_steps_completed", -1)) != 64:
        raise ValueError("completed optimizer steps changed")
    if float(training_metrics.get("adapter_reload_loss_absolute_difference", 1.0)) > 1e-6:
        raise ValueError("adapter reload loss equivalence failed")
    if len(token_audit) != 500:
        raise ValueError("training token audit does not contain 500 candidates")
    if token_budget_audit.get("exposure_gate_passed") is not True:
        raise ValueError("response-token exposure gate failed")
    if int(token_budget_audit.get("optimizer_steps", -1)) != 64:
        raise ValueError("token budget audit optimizer step count changed")
    if len(optimizer_step_rows) != 64:
        raise ValueError("optimizer-step token log does not contain 64 rows")
    observed_steps = [int(row["optimizer_step"]) for row in optimizer_step_rows]
    if observed_steps != list(range(1, 65)):
        raise ValueError("optimizer-step token log is not contiguous")
    exposure = sum(int(row["response_supervision_tokens"]) for row in optimizer_step_rows)
    if exposure != int(token_budget_audit["response_supervision_exposure_tokens"]):
        raise ValueError("optimizer-step token sum differs from token budget audit")
    if exposure != int(training_metrics["supervised_tokens_seen"]):
        raise ValueError("training supervised-token total differs from step log")
    return {
        "selected_count": 500,
        "optimizer_steps_planned": 64,
        "optimizer_steps_completed": 64,
        "response_supervision_exposure_tokens": exposure,
        "token_budget_gate_passed": True,
        "adapter_reload_equivalence_passed": True,
    }


def audit_phase1_run(
    *,
    repo_root: Path,
    config_path: Path,
    cell_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    contract = resolve_phase1_contract(
        repo_root=repo_root,
        config_path=config_path,
        cell_id=cell_id,
    )
    run_dir = run_dir.resolve()
    run_dir.relative_to(contract["output_root"])
    manifest = __import__("json").loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config", {}).get("cell_id") != cell_id:
        raise ValueError("run manifest identifies a different cell")
    if manifest.get("config", {}).get("phase1_config_sha256") != contract["config_sha256"]:
        raise ValueError("run manifest matrix hash changed")
    if manifest.get("config", {}).get("selection_manifest_sha256") != contract[
        "selection"
    ]["file_sha256"]:
        raise ValueError("run manifest selection hash changed")

    training_dir = run_dir / "training_complete"
    training_metrics = __import__("json").loads(
        (training_dir / "training_metrics.json").read_text(encoding="utf-8")
    )
    token_budget = __import__("json").loads(
        (training_dir / "token_budget_audit.json").read_text(encoding="utf-8")
    )
    token_audit = __import__("json").loads(
        (training_dir / "token_audit.json").read_text(encoding="utf-8")
    )
    step_rows = read_jsonl(run_dir / "optimizer_step_tokens.jsonl")
    training_report = audit_training_artifacts(
        training_metrics=training_metrics,
        token_budget_audit=token_budget,
        token_audit=token_audit,
        optimizer_step_rows=step_rows,
    )
    adapter_path = training_dir / "adapter" / "adapter_model.safetensors"
    if file_sha256(adapter_path) != training_metrics.get("adapter_model_sha256"):
        raise ValueError("adapter SHA-256 changed")

    evaluation_dir = run_dir / "evaluation" / "merged"
    raw_path = evaluation_dir / "raw_outputs.jsonl"
    metrics_path = evaluation_dir / "metrics.json"
    rows = read_jsonl(raw_path)
    metrics = __import__("json").loads(metrics_path.read_text(encoding="utf-8"))
    all_records = read_jsonl(contract["data_dir"] / "gsm8k_records.jsonl")
    frozen_records = sorted(
        (row for row in all_records if row["protocol_split"] == "held_out_test"),
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    evaluation_report = audit_completed_evaluation(
        rows=rows,
        frozen_records=frozen_records,
        metrics=metrics,
        prompt_version=contract["config"]["evaluation"]["prompt_version"],
    )
    evaluation_report.pop("recomputed_metrics", None)
    completion = __import__("json").loads(
        (run_dir / "cell_complete.json").read_text(encoding="utf-8")
    )
    if completion.get("status") != "PASS" or completion.get("cell_id") != cell_id:
        raise ValueError("cell completion artifact changed")
    if completion.get("raw_outputs_sha256") != file_sha256(raw_path):
        raise ValueError("completion raw-output hash changed")
    return {
        "audit_schema_version": "budget-equivalent-cell-audit-v3",
        "status": "PASS",
        "cell_id": cell_id,
        "source_run_id": manifest["run_id"],
        "method": contract["method"],
        "replicate_index": contract["replicate_index"],
        "train_seed": contract["seed"],
        "selection_manifest_sha256": contract["selection"]["file_sha256"],
        "selected_id_sha256": contract["selection"]["selected_id_sha256"],
        "training": training_report,
        "evaluation": evaluation_report,
        "artifact_hashes": {
            "adapter_model": file_sha256(adapter_path),
            "training_metrics": file_sha256(training_dir / "training_metrics.json"),
            "token_budget_audit": file_sha256(training_dir / "token_budget_audit.json"),
            "raw_outputs": file_sha256(raw_path),
            "evaluation_metrics": file_sha256(metrics_path),
            "cell_complete": file_sha256(run_dir / "cell_complete.json"),
        },
        "accuracy_withheld": True,
        "next_cell_started": False,
    }
