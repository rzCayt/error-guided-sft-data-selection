"""Blind-safe v4 audit for one completed budget-equivalent Phase 1 cell."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from eg_sft.evaluation.resumable import aggregate_gsm8k_metrics
from eg_sft.experiment.b500_engineering_audit import audit_completed_evaluation
from eg_sft.experiment.budget_equivalent_audit import audit_training_artifacts
from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract
from eg_sft.gsm8k.parser import parse_generated_answer, parse_last_numeric_answer
from eg_sft.training.b500 import file_sha256, read_jsonl


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def recompute_generation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Re-run the frozen parser from raw text without exposing aggregate accuracy."""

    strict = parse_generated_answer(str(row["raw_output"]))
    if strict.ok:
        prediction = strict
        parse_mode = "strict_final_marker"
    else:
        prediction = parse_last_numeric_answer(str(row["raw_output"]))
        parse_mode = "last_numeric_fallback" if prediction.ok else "failed"
    gold = Decimal(str(row["gold_value"]))
    expected = {
        "strict_parse_status": strict.status,
        "strict_parsed_prediction": (
            str(strict.value) if strict.value is not None else None
        ),
        "parse_mode": parse_mode,
        "parse_status": prediction.status,
        "parsed_prediction": (
            str(prediction.value) if prediction.value is not None else None
        ),
        "numeric_correct": bool(prediction.ok and prediction.value == gold),
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"stored parser field changed for {row['record_id']}: {field}")
    return dict(row) | expected


def validate_blind_merged_metrics(
    *, metrics: dict[str, Any], rows: list[dict[str, Any]], raw_path: Path
) -> None:
    """Validate sealed metadata without requiring intentionally withheld accuracy."""

    if metrics.get("status") != "PASS":
        raise ValueError("merged evaluation status is not PASS")
    if metrics.get("accuracy_withheld") is not True:
        raise ValueError("merged evaluation is not accuracy-blind")
    if int(metrics.get("record_count", -1)) != len(rows):
        raise ValueError("merged evaluation record count changed")
    if metrics.get("raw_outputs_sha256") != file_sha256(raw_path):
        raise ValueError("merged evaluation raw-output hash changed")
    if int(metrics.get("worker_count", -1)) != 2:
        raise ValueError("merged evaluation worker count changed")
    workers = metrics.get("workers")
    if not isinstance(workers, list) or sorted(
        int(worker.get("record_count", -1)) for worker in workers
    ) != [659, 660]:
        raise ValueError("merged evaluation worker counts changed")


def audit_phase1_run_v4(
    *, repo_root: Path, config_path: Path, cell_id: str, run_dir: Path
) -> dict[str, Any]:
    contract = resolve_phase1_contract(
        repo_root=repo_root,
        config_path=config_path,
        cell_id=cell_id,
    )
    run_dir = run_dir.resolve()
    run_dir.relative_to(contract["output_root"])
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("config", {}).get("cell_id") != cell_id:
        raise ValueError("run manifest identifies a different cell")
    if manifest.get("config", {}).get("phase1_config_sha256") != contract[
        "config_sha256"
    ]:
        raise ValueError("run manifest matrix hash changed")
    if manifest.get("config", {}).get("selection_manifest_sha256") != contract[
        "selection"
    ]["file_sha256"]:
        raise ValueError("run manifest selection hash changed")

    training_dir = run_dir / "training_complete"
    training_metrics = _read_json(training_dir / "training_metrics.json")
    token_budget = _read_json(training_dir / "token_budget_audit.json")
    token_audit = json.loads(
        (training_dir / "token_audit.json").read_text(encoding="utf-8")
    )
    if not isinstance(token_audit, list):
        raise ValueError("training token audit must contain a JSON list")
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
    sealed_metrics = _read_json(metrics_path)
    validate_blind_merged_metrics(
        metrics=sealed_metrics,
        rows=rows,
        raw_path=raw_path,
    )
    recomputed_rows = [recompute_generation_row(row) for row in rows]
    recomputed_metrics = aggregate_gsm8k_metrics(recomputed_rows)
    frozen_records = sorted(
        (
            row
            for row in read_jsonl(contract["data_dir"] / "gsm8k_records.jsonl")
            if row["protocol_split"] == "held_out_test"
        ),
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    evaluation_report = audit_completed_evaluation(
        rows=recomputed_rows,
        frozen_records=frozen_records,
        metrics=recomputed_metrics,
        prompt_version=contract["config"]["evaluation"]["prompt_version"],
    )
    evaluation_report.pop("recomputed_metrics", None)
    evaluation_report["parser_rows_recomputed_from_raw_text"] = len(rows)
    evaluation_report["sealed_metrics_schema"] = "blind-metadata-only"

    completion_path = run_dir / "cell_complete.json"
    completion = _read_json(completion_path)
    if completion.get("status") != "PASS" or completion.get("cell_id") != cell_id:
        raise ValueError("cell completion artifact changed")
    if completion.get("raw_outputs_sha256") != file_sha256(raw_path):
        raise ValueError("completion raw-output hash changed")
    return {
        "audit_schema_version": "budget-equivalent-cell-audit-v4-blind-safe",
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
            "sealed_evaluation_metrics": file_sha256(metrics_path),
            "cell_complete": file_sha256(completion_path),
        },
        "accuracy_withheld": True,
        "next_cell_started": False,
    }
