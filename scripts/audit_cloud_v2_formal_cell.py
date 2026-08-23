"""Read-only, accuracy-blind integrity audit for one completed formal cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.formal_two_worker import (  # noqa: E402
    formal_shards,
    merge_formal_worker_outputs,
)
from eg_sft.evaluation.gsm8k_generation import score_generation  # noqa: E402
from eg_sft.evaluation.resumable import validate_completed_prefix  # noqa: E402
from eg_sft.experiment.cloud_v2_analysis import _load_valid_checkpoints  # noqa: E402
from eg_sft.experiment.cloud_v2_formal import (  # noqa: E402
    FORMAL_METHODS,
    FORMAL_SEEDS,
    resolve_formal_contract,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud_v2_formal_b500_single_cell_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    method = str(manifest.get("config", {}).get("method", ""))
    seed = int(manifest.get("seed", -1))
    if method not in FORMAL_METHODS or seed not in FORMAL_SEEDS:
        raise ValueError("run manifest is not one frozen formal cell")
    contract = resolve_formal_contract(
        repo_root=ROOT,
        config_path=args.config.resolve(),
        method=method,
        seed=seed,
    )
    run_dir.relative_to(contract["output_root"])
    run_config = manifest["config"]
    checks: dict[str, bool] = {
        "formal_config_hash": run_config.get("formal_config_sha256")
        == contract["config_sha256"],
        "selection_manifest_hash": run_config.get("selection_manifest_sha256")
        == contract["selection"]["file_sha256"],
        "selected_id_hash": run_config.get("selected_id_sha256")
        == contract["selection"]["selected_id_sha256"],
        "git_commit_present": isinstance(manifest.get("git_commit"), str)
        and bool(manifest["git_commit"]),
    }
    training_dir = run_dir / "training_complete"
    training = _read_json(training_dir / "training_metrics.json")
    adapter_path = training_dir / "adapter" / "adapter_model.safetensors"
    checks.update(
        {
            "training_status": training.get("status") == "PASS",
            "training_selected_count": int(training.get("selected_count", -1)) == 500,
            "training_epochs": int(training.get("epochs", -1)) == 2,
            "training_micro_batch": int(training.get("micro_batch_size", -1)) == 1,
            "training_accumulation": int(
                training.get("gradient_accumulation_steps", -1)
            )
            == 16,
            "training_optimizer_steps": int(
                training.get("optimizer_steps_completed", -1)
            )
            == 63,
            "adapter_saved": adapter_path.is_file(),
            "adapter_hash": adapter_path.is_file()
            and file_sha256(adapter_path) == training.get("adapter_model_sha256"),
            "adapter_reload": float(
                training.get("adapter_reload_loss_absolute_difference", 1.0)
            )
            <= 1e-6,
        }
    )
    checkpoints = _load_valid_checkpoints(run_dir)
    initial = [row for row in checkpoints if int(row[0]["optimizer_steps"]) == 0]
    final = [row for row in checkpoints if int(row[0]["optimizer_steps"]) == 63]
    checks["one_initial_checkpoint"] = len(initial) == 1
    checks["one_final_checkpoint"] = len(final) == 1
    checks["final_checkpoint_cursor"] = len(final) == 1 and int(
        final[0][0]["next_micro_batch_index"]
    ) == 1000

    evaluation = contract["config"]["evaluation"]
    shards = formal_shards(evaluation)
    worker_payloads = {}
    for shard in shards:
        worker_dir = run_dir / "evaluation" / "workers" / shard.shard_id
        worker_payloads[shard.shard_id] = {
            "manifest": _read_json(worker_dir / "manifest.json"),
            "metrics": _read_json(worker_dir / "metrics.json"),
            "rows": read_jsonl(worker_dir / "raw_outputs.jsonl"),
        }
    all_records = read_jsonl(contract["data_dir"] / "gsm8k_records.jsonl")
    records = sorted(
        (row for row in all_records if row["protocol_split"] == evaluation["split"]),
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    recomputed_merged, merge_report = merge_formal_worker_outputs(
        frozen_records=records,
        shards=shards,
        worker_payloads=worker_payloads,
    )
    merged_dir = run_dir / "evaluation" / "merged"
    merged_manifest = _read_json(merged_dir / "manifest.json")
    merged_metrics = _read_json(merged_dir / "metrics.json")
    merged_path = merged_dir / "raw_outputs.jsonl"
    merged_rows = read_jsonl(merged_path)
    validate_completed_prefix(completed_rows=merged_rows, frozen_records=records)
    checks.update(
        {
            "worker_merge_status": merge_report.get("status") == "PASS",
            "same_gpu_uuid": merge_report.get("gpu_uuid")
            == merged_manifest.get("gpu_uuid"),
            "same_adapter": merge_report.get("adapter_model_sha256")
            == training.get("adapter_model_sha256"),
            "merged_record_count": len(merged_rows) == 1319,
            "merged_rows_equal_worker_merge": merged_rows == recomputed_merged,
            "merged_raw_hash": file_sha256(merged_path)
            == merged_manifest.get("raw_outputs_sha256")
            == merged_metrics.get("raw_outputs_sha256"),
            "merged_accuracy_withheld": merged_metrics.get("accuracy_withheld") is True,
        }
    )
    gsm_test = load_dataset(
        contract["protocol"]["datasets"]["gsm8k"]["repo_id"],
        contract["protocol"]["datasets"]["gsm8k"]["config"],
        split="test",
        revision=contract["protocol"]["datasets"]["gsm8k"]["revision"],
    )
    row_integrity = True
    compared_fields = (
        "record_id",
        "source_index",
        "question_sha256",
        "prompt_version",
        "raw_output",
        "strict_parse_status",
        "strict_parsed_prediction",
        "parse_mode",
        "parse_status",
        "parsed_prediction",
        "gold_value",
        "numeric_correct",
    )
    for record, row in zip(records, merged_rows, strict=True):
        source_row = gsm_test[int(record["source_index"])]
        validate_gsm8k_source_row(record, source_row)
        rescored = score_generation(
            record=record,
            gold_answer_text=source_row["answer"],
            generated_text=row["raw_output"],
        )
        if any(row.get(field) != rescored.get(field) for field in compared_fields):
            row_integrity = False
            break
    checks["all_rows_reparse_exactly"] = row_integrity
    completion = _read_json(run_dir / "cell_complete.json")
    checks.update(
        {
            "cell_complete_status": completion.get("status") == "PASS",
            "cell_complete_raw_hash": completion.get("raw_outputs_sha256")
            == file_sha256(merged_path),
            "cell_complete_adapter_hash": completion.get("adapter_model_sha256")
            == file_sha256(adapter_path),
            "cell_did_not_start_next": completion.get("next_cell_started") is False,
        }
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "audit_version": "cloud-v2-formal-cell-audit-v1",
        "status": status,
        "run_id": manifest["run_id"],
        "checks": checks,
        "hashes": {
            "formal_config_sha256": contract["config_sha256"],
            "adapter_model_sha256": file_sha256(adapter_path),
            "raw_outputs_sha256": file_sha256(merged_path),
            "cell_complete_sha256": file_sha256(run_dir / "cell_complete.json"),
        },
        "record_count": 1319,
        "accuracy_withheld": True,
        "method_comparison_withheld": True,
    }
    output = args.output.resolve() if args.output else run_dir / "audit" / "formal_cell_audit.json"
    _write_json_exclusive(output, report)
    print(
        json.dumps(
            {
                "status": status,
                "stage": "formal_cell_audit",
                "run_id": manifest["run_id"],
                "hashes": {**report["hashes"], "audit_sha256": file_sha256(output)},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
