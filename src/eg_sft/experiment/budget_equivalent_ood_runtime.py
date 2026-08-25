"""Frozen contracts and CPU-only audits for arithmetic OOD evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eg_sft.data.public_gsm8k import sha256_text
from eg_sft.evaluation.arithmetic_ood import build_ood_record, parse_unique_numeric_gold
from eg_sft.evaluation.gsm8k_generation import PROMPT_VERSION
from eg_sft.experiment.budget_equivalent_matrix import (
    read_json_object,
    resolve_frozen_file,
    validate_matrix_config,
)
from eg_sft.experiment.budget_equivalent_protocol import repository_path
from eg_sft.training.b500 import file_sha256, read_jsonl


OOD_DATASETS = ("svamp", "asdiv_numeric", "multiarith")


def canonical_source_row_sha256(row: dict[str, Any]) -> str:
    """Hash one source row exactly as the frozen OOD manifest builder does."""

    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


def resolve_ood_contract(
    *, repo_root: Path, matrix_config_path: Path, dataset: str
) -> dict[str, Any]:
    """Resolve one text-free OOD membership list and its frozen source revision."""

    if dataset not in OOD_DATASETS:
        raise ValueError(f"unsupported OOD dataset: {dataset}")
    matrix_config_path = matrix_config_path.resolve()
    matrix = read_json_object(matrix_config_path)
    validate_matrix_config(matrix)
    ood = matrix.get("ood_evaluation")
    if not isinstance(ood, dict) or ood.get("required_before_unblinding") is not True:
        raise ValueError("formal OOD binding is missing")
    if ood.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("OOD prompt version changed")

    manifest_path = resolve_frozen_file(
        repo_root=repo_root, binding=ood["manifest"], label="OOD manifest"
    )
    manifest = read_json_object(manifest_path)
    if manifest.get("raw_dataset_text_stored") is not False:
        raise ValueError("OOD manifest unexpectedly contains raw dataset text")

    binding = ood["datasets"][dataset]
    records_path = resolve_frozen_file(
        repo_root=repo_root, binding=binding, label=f"{dataset} OOD records"
    )
    records = read_jsonl(records_path)
    if len(records) != int(binding["expected_record_count"]):
        raise ValueError(f"{dataset} OOD record count changed")
    if any(str(row.get("dataset")) != dataset for row in records):
        raise ValueError(f"{dataset} OOD records contain another dataset")
    if any("question" in row or "answer" in row for row in records):
        raise ValueError("text-free OOD manifest contains raw question or answer")

    ood_config_path = repository_path(repo_root, "configs/budget_equivalent_ood_v1.json")
    if file_sha256(ood_config_path) != manifest.get("ood_config_sha256"):
        raise ValueError("OOD source configuration hash changed")
    ood_config = read_json_object(ood_config_path)
    source = ood_config["datasets"][dataset]
    protocol_path = resolve_frozen_file(
        repo_root=repo_root,
        binding=matrix["protocol_config"],
        label="protocol config",
    )
    return {
        "matrix": matrix,
        "matrix_config_path": matrix_config_path,
        "matrix_config_sha256": file_sha256(matrix_config_path),
        "protocol": read_json_object(protocol_path),
        "dataset": dataset,
        "source": source,
        "records": records,
        "records_path": records_path,
        "records_sha256": file_sha256(records_path),
        "manifest_path": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "prompt_version": PROMPT_VERSION,
    }


def validate_source_row(
    *, record: Mapping[str, Any], raw_row: Mapping[str, Any], answer_field: str
) -> str:
    """Validate source text against the text-free manifest and return numeric gold."""

    raw = dict(raw_row)
    if canonical_source_row_sha256(raw) != record.get("source_row_sha256"):
        raise ValueError(f"source row hash changed: {record.get('record_id')}")
    rebuilt = build_ood_record(
        dataset=str(record["dataset"]),
        source_index=int(record["source_index"]),
        row=raw,
        answer_field=answer_field,
    )
    for field in ("record_id", "question_sha256", "answer_sha256", "gold_parse_status"):
        if rebuilt.get(field) != record.get(field):
            raise ValueError(f"source row {field} changed: {record.get('record_id')}")
    gold, status = parse_unique_numeric_gold(raw.get(answer_field, ""))
    if gold is None or status != "ok":
        raise ValueError(f"source gold is no longer uniquely numeric: {record['record_id']}")
    gold_value = str(gold)
    if sha256_text(gold_value) != record.get("gold_value_sha256"):
        raise ValueError(f"source gold hash changed: {record['record_id']}")
    return gold_value


def contiguous_shard(
    records: Sequence[dict[str, Any]], *, shard_index: int, shard_count: int
) -> tuple[int, int, list[dict[str, Any]]]:
    """Return one deterministic contiguous shard using half-open boundaries."""

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid OOD shard index/count")
    start = len(records) * shard_index // shard_count
    end = len(records) * (shard_index + 1) // shard_count
    return start, end, list(records[start:end])


def validate_resume_worker_manifest(
    *, existing: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Allow resume on a replacement GPU only when UUID is the sole difference."""
    if dict(existing) == dict(expected):
        return False
    old = dict(existing)
    new = dict(expected)
    old_uuid = old.pop("gpu_uuid", None)
    new_uuid = new.pop("gpu_uuid", None)
    if old != new:
        raise ValueError("OOD worker manifest changed beyond GPU UUID")
    if not old_uuid or not new_uuid or old_uuid == new_uuid:
        raise ValueError("OOD worker manifest changed")
    return True


def validate_worker_prefix(
    *, rows: Sequence[Mapping[str, Any]], frozen_records: Sequence[Mapping[str, Any]]
) -> int:
    """Validate a resumable worker prefix without inspecting aggregate accuracy."""

    if len(rows) > len(frozen_records):
        raise ValueError("OOD worker output is longer than its frozen shard")
    for offset, (row, frozen) in enumerate(zip(rows, frozen_records, strict=False)):
        if row.get("record_id") != frozen.get("record_id"):
            raise ValueError(f"OOD record order changed at shard offset {offset}")
        if int(row.get("source_index", -1)) != int(frozen["source_index"]):
            raise ValueError(f"OOD source index changed at shard offset {offset}")
        if row.get("question_sha256") != frozen.get("question_sha256"):
            raise ValueError(f"OOD question hash changed at shard offset {offset}")
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"OOD prompt version changed at shard offset {offset}")
    return len(rows)


def recompute_ood_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute task metrics from immutable row-level outputs."""

    if not rows:
        raise ValueError("cannot aggregate an empty OOD evaluation")
    correct = sum(bool(row.get("numeric_correct")) for row in rows)
    strict = sum(row.get("strict_parse_status") == "ok" for row in rows)
    parsed = sum(row.get("parse_mode") != "failed" for row in rows)
    return {
        "record_count": len(rows),
        "numeric_correct_count": correct,
        "exact_numeric_accuracy": correct / len(rows),
        "strict_parse_count": strict,
        "strict_parse_rate": strict / len(rows),
        "parsed_count": parsed,
        "parse_rate": parsed / len(rows),
    }


def audit_complete_dataset(
    *, rows: Sequence[dict[str, Any]], frozen_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Require exactly one ordered output for every frozen OOD record."""

    if len(rows) != len(frozen_records):
        raise ValueError(
            f"OOD output count changed: {len(rows)} != {len(frozen_records)}"
        )
    validate_worker_prefix(rows=rows, frozen_records=frozen_records)
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("OOD outputs contain duplicate record IDs")
    for row, frozen in zip(rows, frozen_records, strict=True):
        if sha256_text(str(row.get("gold_value"))) != frozen.get("gold_value_sha256"):
            raise ValueError(f"OOD gold value changed: {row['record_id']}")
    return {
        "status": "PASS",
        "record_count": len(rows),
        "unique_record_id_count": len(ids),
        "ordered_frozen_membership": True,
        "gold_hashes_match": True,
        "metrics": recompute_ood_metrics(rows),
    }
