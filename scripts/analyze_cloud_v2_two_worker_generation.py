"""Compare merged same-GPU two-worker output with single-worker batch-one output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.two_worker_calibration import (  # noqa: E402
    compare_to_single_worker_reference,
    validate_two_worker_config,
)
from eg_sft.experiment.cloud_v2_calibration import read_json_object  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-worker-run-dir", type=Path, required=True)
    parser.add_argument("--batch1-reference-run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud_v2_two_worker_generation_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = read_json_object(config_path)
    validate_two_worker_config(config)
    two_worker_dir = args.two_worker_run_dir.resolve()
    reference_dir = args.batch1_reference_run_dir.resolve()
    merged_dir = two_worker_dir / "merged"
    merged_manifest = read_json_object(merged_dir / "manifest.json")
    merged_metrics = read_json_object(merged_dir / "metrics.json")
    reference_manifest = read_json_object(reference_dir / "manifest.json")
    reference_metrics = read_json_object(reference_dir / "metrics.json")
    candidate_rows = read_jsonl(merged_dir / "raw_outputs.jsonl")
    reference_rows = read_jsonl(reference_dir / "raw_outputs.jsonl")
    comparison = compare_to_single_worker_reference(
        reference_rows=reference_rows,
        candidate_rows=candidate_rows,
        max_difference_examples=int(config["comparison_policy"]["max_difference_examples"]),
    )
    reference_adapter = str(reference_manifest.get("config", {}).get("adapter_sha256", ""))
    candidate_adapter = str(merged_manifest.get("adapter_model_sha256", ""))
    integrity_checks = {
        "merged_manifest_pass": merged_manifest.get("status") == "PASS",
        "merged_metrics_pass": merged_metrics.get("status") == "PASS",
        "same_adapter_sha256": (
            bool(reference_adapter) and reference_adapter == candidate_adapter
        ),
        "reference_batch_size_is_one": (
            int(reference_manifest.get("config", {}).get("generation_batch_size", -1)) == 1
        ),
        "reference_metrics_pass": reference_metrics.get("status") == "PASS",
        "merged_raw_hash_matches": (
            file_sha256(merged_dir / "raw_outputs.jsonl")
            == merged_manifest.get("raw_outputs_sha256")
            == merged_metrics.get("raw_outputs_sha256")
        ),
    }
    status = (
        "PASS"
        if comparison["status"] == "PASS" and all(integrity_checks.values())
        else "FAIL"
    )
    report = {
        "analysis_version": "cloud-v2-two-worker-vs-single-worker-v1",
        "status": status,
        "integrity_checks": integrity_checks,
        "record_comparison": comparison,
        "execution_mode": "two_concurrent_batch1_workers_on_one_gpu",
        "not_formal_training": True,
        "end_to_end_wall_seconds": merged_metrics[
            "end_to_end_wall_seconds_this_launcher_invocation"
        ],
        "end_to_end_wall_examples_per_second": merged_metrics[
            "end_to_end_wall_examples_per_second"
        ],
        "throughput_comparable": merged_metrics["throughput_comparable"],
        "gpu_uuid": merged_metrics["gpu_uuid"],
        "workers": merged_metrics["workers"],
        "sum_worker_peak_allocated_memory_gib": merged_metrics[
            "sum_worker_peak_allocated_memory_gib"
        ],
        "sum_worker_peak_reserved_memory_gib": merged_metrics[
            "sum_worker_peak_reserved_memory_gib"
        ],
        "two_worker_config_sha256": file_sha256(config_path),
        "claim_boundary": (
            "A PASS establishes exact development-output equivalence and an engineering "
            "throughput measurement only. It does not justify concurrent formal training."
        ),
    }
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
