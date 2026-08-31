"""Launch exactly two batch-one workers on one CUDA device, then merge safely."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import (  # noqa: E402
    _append_jsonl,
    _git_commit,
    _read_json,
    _require_clean_git_worktree,
    _write_json_exclusive,
)
from run_cloud_v2_generation_worker import (  # noqa: E402
    _load_context,
    _validated_training_artifact,
)

from eg_sft.evaluation.cloud_v2_batching import append_jsonl_rows_fsynced  # noqa: E402
from eg_sft.evaluation.resumable import validate_completed_prefix  # noqa: E402
from eg_sft.evaluation.two_worker_calibration import (  # noqa: E402
    merge_worker_outputs,
    validate_two_worker_config,
)
from eg_sft.experiment.cloud_v2_calibration import repository_path  # noqa: E402
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _enriched_worker_payload(worker_dir: Path) -> dict[str, Any]:
    manifest = _read_json(worker_dir / "manifest.json")
    metrics = _read_json(worker_dir / "metrics.json")
    rows = read_jsonl(worker_dir / "raw_outputs.jsonl")
    failures_path = worker_dir / "failures.jsonl"
    failures = read_jsonl(failures_path) if failures_path.exists() else []
    metrics = dict(metrics)
    metrics["model_load_seconds"] = float(metrics["model_load_seconds"]) + sum(
        float(row.get("model_load_seconds_this_invocation", 0.0)) for row in failures
    )
    metrics["generation_seconds"] = float(metrics["generation_seconds"]) + sum(
        float(row.get("generation_seconds_this_invocation", 0.0)) for row in failures
    )
    metrics["worker_wall_seconds"] = float(metrics["worker_wall_seconds"]) + sum(
        float(row.get("worker_wall_seconds_this_invocation", 0.0)) for row in failures
    )
    metrics["resume_invocation_count"] = int(metrics.get("resume_invocation_count", 1)) + len(
        failures
    )
    metrics["prior_failure_count"] = len(failures)
    return {"manifest": manifest, "metrics": metrics, "rows": rows}


def _create_or_resume_launcher(
    *,
    output_root: Path,
    run_config: dict[str, Any],
    protocol: dict[str, Any],
    training_manifest: dict[str, Any],
    adapter_sha256: str,
    resume_run_dir: Path | None,
) -> tuple[Path, dict[str, Any]]:
    if resume_run_dir is None:
        run_dir, manifest = create_run_manifest(
            output_root=output_root,
            repo_root=ROOT,
            stage="cloud_v2_two_worker_generation",
            config=run_config,
            seed=int(training_manifest["seed"]),
            command=[str(Path(__file__)), "--training-run-dir", training_manifest["run_id"]],
            dataset_revisions={
                protocol["datasets"]["gsm8k"]["repo_id"]: protocol["datasets"]["gsm8k"][
                    "revision"
                ]
            },
            model_revision=protocol["model"]["revision"],
            extra={"gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__},
        )
        launcher_manifest = {
            **manifest,
            "launcher_schema_version": "cloud-v2-two-worker-launcher-v1",
            "adapter_model_sha256": adapter_sha256,
            "execution_mode": "two_concurrent_processes_on_one_gpu_not_formal_training",
        }
        _write_json_exclusive(run_dir / "launcher_manifest.json", launcher_manifest)
        return run_dir, launcher_manifest
    run_dir = resume_run_dir.resolve()
    run_dir.relative_to(output_root)
    manifest = _read_json(run_dir / "launcher_manifest.json")
    if manifest["config"] != run_config:
        raise ValueError("launcher resume contract changed")
    if manifest["git_commit"] != _git_commit():
        raise ValueError("launcher resume requires the original git commit")
    if manifest["adapter_model_sha256"] != adapter_sha256:
        raise ValueError("launcher resume adapter changed")
    return run_dir, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud_v2_two_worker_generation_v1.json"),
    )
    parser.add_argument("--resume-run-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("two-worker launcher requires a BF16 CUDA GPU")
    _require_clean_git_worktree()
    config_path = args.config.resolve()
    context = _load_context(config_path, "shard0")
    config = context["config"]
    shards = validate_two_worker_config(config)
    training_run_dir = args.training_run_dir.resolve()
    training = _validated_training_artifact(
        training_run_dir=training_run_dir,
        base_config=context["base_config"],
    )
    run_config = {
        "study_role": config["study_role"],
        "two_worker_config_sha256": file_sha256(config_path),
        "base_calibration_config_sha256": file_sha256(context["base_config_path"]),
        "protocol_config_sha256": file_sha256(context["protocol_path"]),
        "source_training_run_id": training["manifest"]["run_id"],
        "source_training_config_hash": training["manifest"]["config_hash"],
        "adapter_model_sha256": training["adapter_sha256"],
        "worker_count": 2,
        "physical_batch_size_per_worker": 1,
        "cuda_device_index": int(config["cuda_device_index"]),
        "shards": config["shards"],
    }
    output_root = repository_path(
        ROOT,
        str(config["output_root"]),
        label="two-worker output root",
    )
    run_dir, launcher_manifest = _create_or_resume_launcher(
        output_root=output_root,
        run_config=run_config,
        protocol=context["protocol"],
        training_manifest=training["manifest"],
        adapter_sha256=training["adapter_sha256"],
        resume_run_dir=args.resume_run_dir,
    )
    final_metrics_path = run_dir / "merged" / "metrics.json"
    if final_metrics_path.exists():
        print(final_metrics_path.read_text(encoding="utf-8"))
        return

    launcher_started = time.perf_counter()
    processes: list[tuple[str, subprocess.Popen, Any]] = []
    skipped_completed: list[str] = []
    for shard in shards:
        worker_dir = run_dir / "workers" / shard.shard_id
        worker_metrics = worker_dir / "metrics.json"
        if worker_metrics.exists():
            skipped_completed.append(shard.shard_id)
            continue
        log_path = run_dir / f"{shard.shard_id}.log"
        mode = "a" if log_path.exists() else "x"
        log_handle = log_path.open(mode, encoding="utf-8", newline="\n")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_cloud_v2_generation_worker.py"),
            "--launcher-run-dir",
            str(run_dir),
            "--training-run-dir",
            str(training_run_dir),
            "--config",
            str(config_path),
            "--shard-id",
            shard.shard_id,
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard.shard_id, process, log_handle))
    return_codes: dict[str, int] = {shard_id: 0 for shard_id in skipped_completed}
    for shard_id, process, log_handle in processes:
        return_codes[shard_id] = process.wait()
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()
    failed_workers = {
        shard_id: return_code
        for shard_id, return_code in return_codes.items()
        if return_code != 0
    }
    if failed_workers:
        _append_jsonl(
            run_dir / "launcher_failures.jsonl",
            {
                "event": "worker_process_failure",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "return_codes": return_codes,
                "failed_workers": failed_workers,
                "pass_merge_created": False,
            },
        )
        raise RuntimeError(f"worker failure prevented merge: {failed_workers}")

    all_records = read_jsonl(context["data_manifest_dir"] / "gsm8k_records.jsonl")
    frozen_records = sorted(
        (
            row
            for row in all_records
            if row["protocol_split"] == config["generation_protocol_split"]
        ),
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    worker_payloads = {
        shard.shard_id: _enriched_worker_payload(
            run_dir / "workers" / shard.shard_id
        )
        for shard in shards
    }
    merged_rows, merge_report = merge_worker_outputs(
        frozen_records=frozen_records,
        shards=shards,
        worker_payloads=worker_payloads,
    )
    merged_dir = run_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_raw_path = merged_dir / "raw_outputs.jsonl"
    existing_merged = read_jsonl(merged_raw_path) if merged_raw_path.exists() else []
    next_index = validate_completed_prefix(
        completed_rows=existing_merged,
        frozen_records=frozen_records,
    )
    append_jsonl_rows_fsynced(merged_raw_path, merged_rows[next_index:])
    completed_merged = read_jsonl(merged_raw_path)
    validate_completed_prefix(
        completed_rows=completed_merged,
        frozen_records=frozen_records,
    )
    if len(completed_merged) != int(config["expected_record_count"]):
        raise ValueError("merged artifact is incomplete")
    end_to_end_wall_seconds = time.perf_counter() - launcher_started
    prior_failures = sum(
        int(payload["metrics"].get("prior_failure_count", 0))
        for payload in worker_payloads.values()
    )
    final_metrics = {
        **merge_report,
        "run_id": launcher_manifest["run_id"],
        "end_to_end_wall_seconds_this_launcher_invocation": end_to_end_wall_seconds,
        "end_to_end_wall_examples_per_second": len(completed_merged)
        / end_to_end_wall_seconds,
        "resume_launcher_used": args.resume_run_dir is not None,
        "prior_worker_failure_count": prior_failures,
        "throughput_comparable": (
            args.resume_run_dir is None and prior_failures == 0 and not skipped_completed
        ),
        "raw_outputs_sha256": file_sha256(merged_raw_path),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "Same-GPU two-process generation calibration only; this is not two formal "
            "training jobs and not a model-quality result."
        ),
    }
    _write_json_exclusive(merged_dir / "metrics.json", final_metrics)
    _write_json_exclusive(
        merged_dir / "manifest.json",
        {
            "status": "PASS",
            "launcher_run_id": launcher_manifest["run_id"],
            "two_worker_config_sha256": file_sha256(config_path),
            "adapter_model_sha256": training["adapter_sha256"],
            "gpu_uuid": merge_report["gpu_uuid"],
            "raw_outputs_sha256": file_sha256(merged_raw_path),
        },
    )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
