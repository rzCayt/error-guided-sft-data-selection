"""Read-only integrity audit for one budget-equivalent Phase 1 cell."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eg_sft.experiment.b500_engineering_audit import audit_completed_evaluation
from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract
from eg_sft.training.b500 import file_sha256, read_jsonl


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
