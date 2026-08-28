"""Prevalidate every OOD dataset before resumably publishing masked audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import _git_commit  # noqa: E402

from eg_sft.experiment.budget_equivalent_ood_audit_v2 import (  # noqa: E402
    audit_complete_ood_dataset_from_raw,
)
from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    canonical_jsonl_bytes,
    masked_metrics,
    write_bytes_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_ood_runtime import (  # noqa: E402
    OOD_DATASETS,
    contiguous_shard,
    resolve_ood_contract,
    validate_worker_prefix,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402
from eg_sft.evaluation.identifiable_batch_backend import (  # noqa: E402
    validate_phase2_generation_evidence,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _failure_event_count(path: Path) -> int:
    return len(read_jsonl(path)) if path.is_file() else 0


def _prevalidate_dataset(
    *, config_path: Path, run_dir: Path, dataset: str, shard_count: int
) -> dict[str, Any]:
    contract = resolve_ood_contract(
        repo_root=ROOT,
        matrix_config_path=config_path,
        dataset=dataset,
    )
    combined: list[dict[str, Any]] = []
    worker_hashes = []
    total_failure_events = 0
    for shard_index in range(shard_count):
        _, _, frozen_shard = contiguous_shard(
            contract["records"],
            shard_index=shard_index,
            shard_count=shard_count,
        )
        worker_name = f"shard_{shard_index:02d}_of_{shard_count:02d}"
        worker_dir = run_dir / "evaluation" / "ood" / dataset / "workers" / worker_name
        manifest = _read_json(worker_dir / "manifest.json")
        worker = manifest.get("worker", {})
        if (
            worker.get("dataset") != dataset
            or int(worker.get("shard_index", -1)) != shard_index
            or int(worker.get("shard_count", -1)) != shard_count
            or manifest.get("matrix_config_sha256") != contract["matrix_config_sha256"]
            or manifest.get("ood_records_sha256") != contract["records_sha256"]
        ):
            raise ValueError(f"OOD worker contract changed: {dataset}/{worker_name}")
        raw_path = worker_dir / "raw_outputs.jsonl"
        metrics_path = worker_dir / "metrics.json"
        rows = read_jsonl(raw_path)
        validate_worker_prefix(rows=rows, frozen_records=frozen_shard)
        if len(rows) != len(frozen_shard):
            raise ValueError(f"OOD shard is incomplete: {dataset}/{worker_name}")
        metrics = _read_json(metrics_path)
        if (
            metrics.get("status") != "PASS"
            or int(metrics.get("record_count", -1)) != len(rows)
            or metrics.get("raw_outputs_sha256") != file_sha256(raw_path)
            or metrics.get("accuracy_withheld") is not True
        ):
            raise ValueError(f"OOD worker metrics changed: {dataset}/{worker_name}")
        failures = _failure_event_count(worker_dir / "failures.jsonl")
        total_failure_events += failures
        combined.extend(rows)
        worker_hashes.append(
            {
                "shard_index": shard_index,
                "raw_outputs_sha256": file_sha256(raw_path),
                "metrics_sha256": file_sha256(metrics_path),
                "recorded_failure_event_count": failures,
            }
        )

    if _read_json(config_path).get("matrix_version") == "phase2-crossed-48cell-v7":
        validate_phase2_generation_evidence(combined, eos_token_id=151643)
    report = audit_complete_ood_dataset_from_raw(
        rows=combined,
        frozen_records=contract["records"],
    )
    return {
        "dataset": dataset,
        "combined": combined,
        "masked_metrics": masked_metrics(report),
        "summary": {
            "status": "PASS",
            "dataset": dataset,
            "record_count": int(report["record_count"]),
            "unique_record_id_count": int(report["unique_record_id_count"]),
            "parser_rows_recomputed_from_raw_text": int(
                report["parser_rows_recomputed_from_raw_text"]
            ),
            "ordered_frozen_membership": True,
            "gold_hashes_match": True,
            "worker_count": shard_count,
            "worker_hashes": worker_hashes,
            "recorded_failure_event_count": total_failure_events,
            "recovered_after_recorded_failure": total_failure_events > 0,
            "accuracy_withheld": True,
        },
    }


def _publish_dataset(*, run_dir: Path, validated: dict[str, Any]) -> dict[str, Any]:
    dataset = str(validated["dataset"])
    merged_dir = run_dir / "evaluation" / "ood" / dataset / "merged"
    raw_output = merged_dir / "raw_outputs.jsonl"
    metrics_output = merged_dir / "sealed_metrics.json"
    write_bytes_exclusive_or_verify(
        raw_output,
        canonical_jsonl_bytes(validated["combined"]),
    )
    write_bytes_exclusive_or_verify(
        metrics_output,
        canonical_json_bytes(validated["masked_metrics"]),
    )
    return dict(validated["summary"]) | {
        "merged_raw_outputs_sha256": file_sha256(raw_output),
        "sealed_metrics_sha256": file_sha256(metrics_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()

    # Phase A: fail closed before publishing any merged or final artifact.
    validated = [
        _prevalidate_dataset(
            config_path=config_path,
            run_dir=run_dir,
            dataset=dataset,
            shard_count=args.shard_count,
        )
        for dataset in OOD_DATASETS
    ]
    # Phase B: idempotently materialize exactly the prevalidated evidence.
    datasets = [_publish_dataset(run_dir=run_dir, validated=row) for row in validated]
    final = {
        "audit_schema_version": "budget-equivalent-ood-audit-v3-prevalidated-masked",
        "status": "PASS",
        "audit_code_commit": _git_commit(),
        "matrix_config_sha256": file_sha256(config_path),
        "datasets": datasets,
        "dataset_count": len(datasets),
        "total_record_count": sum(int(row["record_count"]) for row in datasets),
        "total_parser_rows_recomputed": sum(
            int(row["parser_rows_recomputed_from_raw_text"]) for row in datasets
        ),
        "recorded_failure_event_count": sum(
            int(row["recorded_failure_event_count"]) for row in datasets
        ),
        "accuracy_withheld": True,
        "gpu_accessed": False,
    }
    output = run_dir / "audit" / "ood_audit.json"
    write_bytes_exclusive_or_verify(output, canonical_json_bytes(final))
    sha_output = output.with_suffix(".sha256")
    write_bytes_exclusive_or_verify(
        sha_output,
        f"{file_sha256(output)}  {output.name}\n".encode("ascii"),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "budget_equivalent_ood_audit_v3",
                "audit_code_commit": final["audit_code_commit"],
                "dataset_count": len(datasets),
                "total_record_count": final["total_record_count"],
                "total_parser_rows_recomputed": final[
                    "total_parser_rows_recomputed"
                ],
                "recorded_failure_event_count": final[
                    "recorded_failure_event_count"
                ],
                "audit_sha256": file_sha256(output),
                "accuracy_withheld": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
