"""Merge and CPU-audit all frozen arithmetic OOD outputs for one Phase 1 cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_ood_runtime import (  # noqa: E402
    OOD_DATASETS,
    audit_complete_dataset,
    contiguous_shard,
    resolve_ood_contract,
    validate_worker_prefix,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _audit_dataset(
    *, config_path: Path, run_dir: Path, dataset: str, shard_count: int
) -> dict[str, Any]:
    contract = resolve_ood_contract(
        repo_root=ROOT,
        matrix_config_path=config_path,
        dataset=dataset,
    )
    combined: list[dict[str, Any]] = []
    worker_hashes = []
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
        combined.extend(rows)
        worker_hashes.append(
            {
                "shard_index": shard_index,
                "raw_outputs_sha256": file_sha256(raw_path),
                "metrics_sha256": file_sha256(metrics_path),
            }
        )

    report = audit_complete_dataset(
        rows=combined,
        frozen_records=contract["records"],
    )
    merged_dir = run_dir / "evaluation" / "ood" / dataset / "merged"
    raw_output = merged_dir / "raw_outputs.jsonl"
    metrics_output = merged_dir / "sealed_metrics.json"
    _write_jsonl_exclusive(raw_output, combined)
    _write_json_exclusive(metrics_output, report["metrics"] | {"accuracy_withheld": True})
    return {
        "status": "PASS",
        "dataset": dataset,
        "record_count": report["record_count"],
        "unique_record_id_count": report["unique_record_id_count"],
        "ordered_frozen_membership": True,
        "gold_hashes_match": True,
        "worker_count": shard_count,
        "worker_hashes": worker_hashes,
        "merged_raw_outputs_sha256": file_sha256(raw_output),
        "sealed_metrics_sha256": file_sha256(metrics_output),
        "accuracy_withheld": True,
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
    datasets = [
        _audit_dataset(
            config_path=config_path,
            run_dir=run_dir,
            dataset=dataset,
            shard_count=args.shard_count,
        )
        for dataset in OOD_DATASETS
    ]
    final = {
        "audit_schema_version": "budget-equivalent-ood-audit-v1",
        "status": "PASS",
        "matrix_config_sha256": file_sha256(config_path),
        "datasets": datasets,
        "dataset_count": len(datasets),
        "total_record_count": sum(int(row["record_count"]) for row in datasets),
        "accuracy_withheld": True,
        "gpu_accessed": False,
    }
    output = run_dir / "audit" / "ood_audit.json"
    _write_json_exclusive(output, final)
    sidecar = output.with_suffix(".sha256")
    sidecar.write_text(f"{file_sha256(output)}  {output.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "budget_equivalent_ood_audit",
                "dataset_count": len(datasets),
                "total_record_count": final["total_record_count"],
                "audit_sha256": file_sha256(output),
                "accuracy_withheld": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
