"""Integrity primitives for same-GPU, two-worker generation calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TWO_WORKER_VERSION = "cloud-v2-two-worker-generation-v1"
COMPARISON_FIELDS = (
    "raw_output",
    "parse_status",
    "parsed_prediction",
    "numeric_correct",
)


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    start_index: int
    end_index: int

    @property
    def count(self) -> int:
        return self.end_index - self.start_index


def validate_two_worker_config(payload: dict[str, Any]) -> tuple[ShardSpec, ShardSpec]:
    if payload.get("calibration_version") != TWO_WORKER_VERSION:
        raise ValueError("unexpected two-worker calibration version")
    if int(payload.get("worker_count", 0)) != 2:
        raise ValueError("two-worker calibration requires exactly two workers")
    if int(payload.get("physical_batch_size_per_worker", 0)) != 1:
        raise ValueError("each worker must keep physical batch size one")
    if int(payload.get("expected_record_count", 0)) != 128:
        raise ValueError("two-worker calibration requires exactly 128 records")
    if int(payload.get("cuda_device_index", -1)) < 0:
        raise ValueError("cuda_device_index must be non-negative")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != 2:
        raise ValueError("two deterministic shard objects are required")
    shards = tuple(
        ShardSpec(
            shard_id=str(row["shard_id"]),
            start_index=int(row["start_index"]),
            end_index=int(row["end_index"]),
        )
        for row in raw_shards
    )
    if tuple(shard.shard_id for shard in shards) != ("shard0", "shard1"):
        raise ValueError("shard IDs must be shard0 and shard1 in order")
    if (
        shards[0].start_index != 0
        or shards[0].end_index != 64
        or shards[1].start_index != 64
        or shards[1].end_index != 128
    ):
        raise ValueError("frozen shards must be [0,64) and [64,128)")
    return shards


def records_for_shard(
    records: Sequence[dict[str, Any]],
    shard: ShardSpec,
) -> list[dict[str, Any]]:
    if len(records) != 128:
        raise ValueError("frozen development records must contain exactly 128 rows")
    return list(records[shard.start_index : shard.end_index])


def validate_worker_prefix(
    *,
    rows: Sequence[dict[str, Any]],
    frozen_shard_records: Sequence[dict[str, Any]],
    shard_id: str,
) -> int:
    if len(rows) > len(frozen_shard_records):
        raise ValueError(f"{shard_id} output exceeds its frozen range")
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
            raise ValueError(
                f"{shard_id} output is not its frozen prefix at offset {offset}: "
                f"{record_id} != {expected}"
            )
        if row.get("shard_id") not in {None, shard_id}:
            raise ValueError(f"{shard_id} row carries a different shard_id")
    return len(rows)


def merge_worker_outputs(
    *,
    frozen_records: Sequence[dict[str, Any]],
    shards: Sequence[ShardSpec],
    worker_payloads: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_shards = {shard.shard_id for shard in shards}
    if set(worker_payloads) != expected_shards:
        raise ValueError("worker payloads do not contain both frozen shard IDs")
    merged: list[dict[str, Any]] = []
    worker_reports: list[dict[str, Any]] = []
    adapter_hashes: set[str] = set()
    gpu_uuids: set[str] = set()
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
        worker = manifest.get("worker")
        if not isinstance(worker, dict):
            raise ValueError(f"{shard.shard_id} manifest has no worker binding")
        if (
            worker.get("shard_id") != shard.shard_id
            or int(worker.get("start_index", -1)) != shard.start_index
            or int(worker.get("end_index", -1)) != shard.end_index
            or int(worker.get("physical_batch_size", -1)) != 1
        ):
            raise ValueError(f"{shard.shard_id} manifest changed its frozen range")
        frozen_shard = records_for_shard(frozen_records, shard)
        validate_worker_prefix(
            rows=rows,
            frozen_shard_records=frozen_shard,
            shard_id=shard.shard_id,
        )
        if len(rows) != shard.count:
            raise ValueError(f"{shard.shard_id} is missing completed records")
        adapter_hash = str(metrics.get("adapter_model_sha256", ""))
        gpu_uuid = str(metrics.get("gpu_uuid", ""))
        if not adapter_hash or not gpu_uuid:
            raise ValueError(f"{shard.shard_id} omitted adapter hash or GPU UUID")
        adapter_hashes.add(adapter_hash)
        gpu_uuids.add(gpu_uuid)
        merged.extend(rows)
        worker_reports.append(
            {
                "shard_id": shard.shard_id,
                "record_count": len(rows),
                "model_load_seconds": float(metrics["model_load_seconds"]),
                "generation_seconds": float(metrics["generation_seconds"]),
                "worker_wall_seconds": float(metrics["worker_wall_seconds"]),
                "peak_allocated_memory_gib": metrics.get("peak_allocated_memory_gib"),
                "peak_reserved_memory_gib": metrics.get("peak_reserved_memory_gib"),
                "gpu_uuid": gpu_uuid,
                "adapter_model_sha256": adapter_hash,
                "resume_invocation_count": int(metrics.get("resume_invocation_count", 1)),
            }
        )
    if len(adapter_hashes) != 1:
        raise ValueError("workers did not load the same adapter")
    if len(gpu_uuids) != 1:
        raise ValueError("workers did not use the same physical GPU UUID")
    merged_ids = [str(row["record_id"]) for row in merged]
    frozen_ids = [str(row["record_id"]) for row in frozen_records]
    if len(merged_ids) != 128:
        raise ValueError("merged output does not contain exactly 128 rows")
    if len(set(merged_ids)) != 128:
        raise ValueError("merged output contains duplicate record IDs")
    if merged_ids != frozen_ids:
        raise ValueError("merged output is not in the original frozen record order")
    return merged, {
        "status": "PASS",
        "worker_count": 2,
        "physical_batch_size_per_worker": 1,
        "execution_mode": "two_concurrent_processes_on_one_gpu_not_formal_training",
        "record_count": len(merged),
        "adapter_model_sha256": next(iter(adapter_hashes)),
        "gpu_uuid": next(iter(gpu_uuids)),
        "workers": worker_reports,
        "sum_worker_peak_allocated_memory_gib": sum(
            float(row["peak_allocated_memory_gib"] or 0.0) for row in worker_reports
        ),
        "sum_worker_peak_reserved_memory_gib": sum(
            float(row["peak_reserved_memory_gib"] or 0.0) for row in worker_reports
        ),
    }


def compare_to_single_worker_reference(
    *,
    reference_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    max_difference_examples: int = 20,
) -> dict[str, Any]:
    reference_ids = [str(row.get("record_id", "")) for row in reference_rows]
    candidate_ids = [str(row.get("record_id", "")) for row in candidate_rows]
    order_matches = reference_ids == candidate_ids
    unique_candidate_ids = len(set(candidate_ids)) == len(candidate_ids)
    same_id_set = set(reference_ids) == set(candidate_ids)
    reference_by_id = {str(row["record_id"]): row for row in reference_rows}
    candidate_by_id = {str(row["record_id"]): row for row in candidate_rows}
    counts = {field: 0 for field in COMPARISON_FIELDS}
    examples: list[dict[str, Any]] = []
    if unique_candidate_ids and same_id_set:
        for record_id in reference_ids:
            left = reference_by_id[record_id]
            right = candidate_by_id[record_id]
            changed = [field for field in COMPARISON_FIELDS if left.get(field) != right.get(field)]
            for field in changed:
                counts[field] += 1
            if changed and len(examples) < max_difference_examples:
                examples.append(
                    {
                        "record_id": record_id,
                        "changed_fields": changed,
                        "reference": {field: left.get(field) for field in changed},
                        "candidate": {field: right.get(field) for field in changed},
                    }
                )
    exact = (
        len(reference_rows) == len(candidate_rows) == 128
        and order_matches
        and unique_candidate_ids
        and same_id_set
        and all(value == 0 for value in counts.values())
    )
    return {
        "status": "PASS" if exact else "FAIL",
        "record_count_matches": len(reference_rows) == len(candidate_rows) == 128,
        "record_order_matches": order_matches,
        "record_ids_unique": unique_candidate_ids,
        "record_id_set_matches": same_id_set,
        "field_difference_counts": counts,
        "difference_examples": examples,
        "exactly_equivalent": exact,
    }
