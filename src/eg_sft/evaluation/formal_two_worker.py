"""Accuracy-blind integrity helpers for legacy and qualified v4 formal workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eg_sft.evaluation.identifiable_batch_backend import ALLOWED_EVAL_BATCH_SIZES


@dataclass(frozen=True)
class FormalEvalShard:
    shard_id: str
    start_index: int
    end_index: int

    @property
    def count(self) -> int:
        return self.end_index - self.start_index


def formal_shards(evaluation: dict[str, Any]) -> tuple[FormalEvalShard, FormalEvalShard]:
    if int(evaluation.get("worker_count", 0)) != 2:
        raise ValueError("formal evaluation requires two workers")
    if int(evaluation.get("physical_batch_size_per_worker", 0)) != 1:
        raise ValueError("formal workers must keep physical batch size one")
    shards = tuple(
        FormalEvalShard(
            shard_id=str(row["shard_id"]),
            start_index=int(row["start_index"]),
            end_index=int(row["end_index"]),
        )
        for row in evaluation["shards"]
    )
    expected = (
        FormalEvalShard("test_shard0", 0, 660),
        FormalEvalShard("test_shard1", 660, 1319),
    )
    if shards != expected:
        raise ValueError("formal test shards changed")
    return shards


def records_for_formal_shard(
    records: Sequence[dict[str, Any]],
    shard: FormalEvalShard,
) -> list[dict[str, Any]]:
    if len(records) != 1319:
        raise ValueError("formal test records must contain exactly 1319 rows")
    return list(records[shard.start_index : shard.end_index])


def validate_formal_worker_prefix(
    *,
    rows: Sequence[dict[str, Any]],
    frozen_shard_records: Sequence[dict[str, Any]],
    shard_id: str,
) -> int:
    if len(rows) > len(frozen_shard_records):
        raise ValueError(f"{shard_id} exceeds its frozen range")
    seen: set[str] = set()
    for offset, row in enumerate(rows):
        record_id = str(row.get("record_id", ""))
        if not record_id:
            raise ValueError(f"{shard_id} row {offset} has no record_id")
        if record_id in seen:
            raise ValueError(f"{shard_id} contains duplicate record_id {record_id}")
        seen.add(record_id)
        expected = str(frozen_shard_records[offset]["record_id"])
        if record_id != expected:
            raise ValueError(f"{shard_id} is not a frozen prefix at offset {offset}")
        if row.get("shard_id") not in {None, shard_id}:
            raise ValueError(f"{shard_id} row carries a different shard binding")
    return len(rows)


def merge_formal_worker_outputs(
    *,
    frozen_records: Sequence[dict[str, Any]],
    shards: Sequence[FormalEvalShard],
    worker_payloads: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(worker_payloads) != {shard.shard_id for shard in shards}:
        raise ValueError("both formal worker payloads are required")
    merged: list[dict[str, Any]] = []
    adapter_hashes: set[str] = set()
    gpu_uuids: set[str] = set()
    physical_batch_sizes: set[int] = set()
    reports = []
    for shard in shards:
        payload = worker_payloads[shard.shard_id]
        manifest = payload.get("manifest")
        metrics = payload.get("metrics")
        rows = payload.get("rows")
        if not isinstance(manifest, dict) or not isinstance(metrics, dict) or not isinstance(
            rows, list
        ):
            raise ValueError(f"{shard.shard_id} payload is incomplete")
        if metrics.get("status") != "PASS":
            raise ValueError(f"{shard.shard_id} did not finish with PASS")
        worker = manifest.get("worker", {})
        physical_batch_size = int(worker.get("physical_batch_size", -1))
        if (
            worker.get("shard_id") != shard.shard_id
            or int(worker.get("start_index", -1)) != shard.start_index
            or int(worker.get("end_index", -1)) != shard.end_index
            or physical_batch_size not in ALLOWED_EVAL_BATCH_SIZES
        ):
            raise ValueError(f"{shard.shard_id} worker binding changed")
        physical_batch_sizes.add(physical_batch_size)
        frozen_shard = records_for_formal_shard(frozen_records, shard)
        validate_formal_worker_prefix(
            rows=rows,
            frozen_shard_records=frozen_shard,
            shard_id=shard.shard_id,
        )
        if len(rows) != shard.count:
            raise ValueError(f"{shard.shard_id} is incomplete")
        adapter_hash = str(metrics.get("adapter_model_sha256", ""))
        gpu_uuid = str(metrics.get("gpu_uuid", ""))
        if not adapter_hash or not gpu_uuid:
            raise ValueError("formal worker omitted adapter hash or GPU UUID")
        adapter_hashes.add(adapter_hash)
        gpu_uuids.add(gpu_uuid)
        merged.extend(rows)
        reports.append(
            {
                "shard_id": shard.shard_id,
                "record_count": len(rows),
                "raw_outputs_sha256": metrics["raw_outputs_sha256"],
                "model_load_seconds": metrics["model_load_seconds"],
                "generation_seconds": metrics["generation_seconds"],
                "peak_allocated_memory_gib": metrics["peak_allocated_memory_gib"],
                "peak_reserved_memory_gib": metrics["peak_reserved_memory_gib"],
                "gpu_uuid": gpu_uuid,
            }
        )
    if len(adapter_hashes) != 1:
        raise ValueError("formal workers loaded different adapters")
    if len(gpu_uuids) != 1:
        raise ValueError("formal workers used different physical GPU UUIDs")
    if len(physical_batch_sizes) != 1:
        raise ValueError("formal workers used different physical batch sizes")
    merged_ids = [str(row["record_id"]) for row in merged]
    frozen_ids = [str(row["record_id"]) for row in frozen_records]
    if len(merged_ids) != 1319 or len(set(merged_ids)) != 1319:
        raise ValueError("formal merge has missing or duplicate records")
    if merged_ids != frozen_ids:
        raise ValueError("formal merge changed frozen test order")
    return merged, {
        "status": "PASS",
        "record_count": 1319,
        "worker_count": 2,
        "physical_batch_size_per_worker": next(iter(physical_batch_sizes)),
        "adapter_model_sha256": next(iter(adapter_hashes)),
        "gpu_uuid": next(iter(gpu_uuids)),
        "workers": reports,
        "accuracy_withheld": True,
    }
