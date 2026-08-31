"""Pure contracts for the legacy natural-batch1 Phase-2 canary."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CANARY_LEVELS = (
    "raw_ids",
    "first_eos_ids",
    "decoded_text",
    "parsed_number",
    "correctness",
    "strict_status",
)
SEMANTIC_CANARY_LEVELS = (
    "decoded_text",
    "parsed_number",
    "correctness",
    "strict_status",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_exclusive_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_token_ids(token_ids: Sequence[int], eos_token_id: int) -> list[int]:
    values = [int(value) for value in token_ids]
    try:
        end = values.index(int(eos_token_id)) + 1
    except ValueError:
        end = len(values)
    return values[:end]


def _raw_ids(row: Mapping[str, Any]) -> list[int]:
    for field in ("raw_generated_tensor_ids", "generated_token_ids", "raw_ids"):
        values = row.get(field)
        if isinstance(values, list):
            return [int(value) for value in values]
    raise ValueError(f"row has no generated token IDs: {row.get('record_id')}")


def canary_signature(
    row: Mapping[str, Any], *, eos_token_id: int
) -> dict[str, Any]:
    raw_ids = _raw_ids(row)
    canonical_ids = canonical_token_ids(raw_ids, eos_token_id)
    return {
        "record_id": str(row["record_id"]),
        "source_index": int(row["source_index"]),
        "question_sha256": str(row["question_sha256"]),
        "prompt_version": str(row["prompt_version"]),
        "raw_ids": raw_ids,
        "first_eos_ids": canonical_ids,
        "decoded_text": str(row["raw_output"]),
        "parsed_number": row.get("parsed_prediction"),
        "correctness": bool(row.get("numeric_correct")),
        "strict_status": str(row["strict_parse_status"]),
        "parse_mode": str(row["parse_mode"]),
        "parse_status": str(row["parse_status"]),
        "gold_value": str(row["gold_value"]),
    }


def semantic_canary_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    """Create the strongest reference available from historical rows without IDs."""

    return {
        "record_id": str(row["record_id"]),
        "source_index": int(row["source_index"]),
        "question_sha256": str(row["question_sha256"]),
        "prompt_version": str(row["prompt_version"]),
        "decoded_text": str(row["raw_output"]),
        "parsed_number": row.get("parsed_prediction"),
        "correctness": bool(row.get("numeric_correct")),
        "strict_status": str(row["strict_parse_status"]),
        "parse_mode": str(row["parse_mode"]),
        "parse_status": str(row["parse_status"]),
        "gold_value": str(row["gold_value"]),
    }


def select_canary_signatures(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    selected_record_ids: Sequence[str],
    eos_token_id: int,
) -> list[dict[str, Any]]:
    by_id = {str(row["record_id"]): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("source rows contain duplicate record IDs")
    selected = [str(value) for value in selected_record_ids]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("canary requires exactly 16 unique record IDs")
    missing = [record_id for record_id in selected if record_id not in by_id]
    if missing:
        raise ValueError(f"canary source is missing records: {missing}")
    return [
        canary_signature(by_id[record_id], eos_token_id=eos_token_id)
        for record_id in selected
    ]


def select_semantic_canary_signatures(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    selected_record_ids: Sequence[str],
) -> list[dict[str, Any]]:
    by_id = {str(row["record_id"]): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("source rows contain duplicate record IDs")
    selected = [str(value) for value in selected_record_ids]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("canary requires exactly 16 unique record IDs")
    missing = [record_id for record_id in selected if record_id not in by_id]
    if missing:
        raise ValueError(f"canary source is missing records: {missing}")
    return [semantic_canary_signature(by_id[record_id]) for record_id in selected]


def compare_canary_signatures(
    *,
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    comparison_levels: Sequence[str] = CANARY_LEVELS,
) -> dict[str, Any]:
    if len(reference) != 16 or len(candidate) != 16:
        raise ValueError("canary comparison requires two complete 16-row sets")
    reference_ids = [str(row["record_id"]) for row in reference]
    candidate_ids = [str(row["record_id"]) for row in candidate]
    if reference_ids != candidate_ids:
        raise ValueError("canary record order changed")
    levels = tuple(str(level) for level in comparison_levels)
    if not levels or any(level not in CANARY_LEVELS for level in levels):
        raise ValueError("unknown or empty canary comparison levels")
    mismatch_ids = {level: [] for level in levels}
    for expected, observed in zip(reference, candidate, strict=True):
        record_id = str(expected["record_id"])
        for level in levels:
            if expected.get(level) != observed.get(level):
                mismatch_ids[level].append(record_id)
        for identity in (
            "source_index",
            "question_sha256",
            "prompt_version",
            "gold_value",
            "parse_mode",
            "parse_status",
        ):
            if expected.get(identity) != observed.get(identity):
                raise ValueError(f"canary identity field changed: {record_id}/{identity}")
    mismatch_counts = {level: len(values) for level, values in mismatch_ids.items()}
    passed = all(value == 0 for value in mismatch_counts.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "record_count": 16,
        "comparison_levels": list(levels),
        "mismatch_counts": mismatch_counts,
        "mismatch_record_ids": mismatch_ids,
        "exact_all_levels": passed,
    }


def validate_reference_manifest(
    *, manifest: Mapping[str, Any], reference_path: Path
) -> None:
    if manifest.get("schema_version") != "phase2-v7-canary-reference-v1":
        raise ValueError("unexpected canary reference schema")
    if int(manifest.get("record_count", -1)) != 16:
        raise ValueError("canary reference count changed")
    if manifest.get("reference_sha256") != file_sha256(reference_path):
        raise ValueError("canary reference SHA-256 changed")
    if tuple(manifest.get("comparison_levels", ())) != CANARY_LEVELS:
        raise ValueError("canary comparison levels changed")
    if manifest.get("fresh_process_required") is not True:
        raise ValueError("canary must require a fresh process")
    if manifest.get("fresh_output_path_required") is not True:
        raise ValueError("canary must require a fresh output path")


def validate_canary_audit(
    *, audit: Mapping[str, Any], expected_role: str, environment_contract_sha256: str
) -> None:
    if audit.get("schema_version") != "phase2-v7-canary-audit-v1":
        raise ValueError("unexpected canary audit schema")
    if audit.get("role") != expected_role:
        raise ValueError("canary audit role changed")
    if audit.get("status") != "PASS" or audit.get("exact_all_levels") is not True:
        raise ValueError(f"{expected_role} canary did not pass")
    if int(audit.get("record_count", -1)) != 16:
        raise ValueError("canary audit record count changed")
    if audit.get("environment_contract_sha256") != environment_contract_sha256:
        raise ValueError("canary environment contract changed")


def validate_legacy_backend_report(
    *, report_path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise ValueError("legacy backend report does not exist")
    if expected_sha256 is not None and file_sha256(report_path) != expected_sha256:
        raise ValueError("legacy backend report SHA-256 changed")
    report = read_json(report_path)
    if report.get("schema_version") != "phase2-v7-legacy-backend-validation-v1":
        raise ValueError("unexpected legacy backend report schema")
    if report.get("status") != "LEGACY_BATCH1_VALIDATED":
        raise ValueError("legacy batch1 backend has not been validated")
    backend = report.get("eval_backend", {})
    required = {
        "batch_size": 1,
        "padding_policy": "natural_per_example",
        "do_sample": False,
        "num_beams": 1,
        "max_input_tokens": 512,
        "max_new_tokens": 256,
        "dtype": "bf16",
        "attention_backend": "sdpa",
        "batch_gt1_authorized": False,
    }
    for field, expected in required.items():
        if backend.get(field) != expected:
            raise ValueError(f"legacy backend field changed: {field}")
    environment_sha = str(report.get("environment_contract_sha256", ""))
    if len(environment_sha) != 64:
        raise ValueError("legacy backend environment SHA is invalid")
    for role in ("base_model_16", "archived_adapter_16"):
        audit = report.get("canaries", {}).get(role)
        if not isinstance(audit, dict):
            raise ValueError(f"missing canary audit: {role}")
        validate_canary_audit(
            audit=audit,
            expected_role=role,
            environment_contract_sha256=environment_sha,
        )
    if report.get("batch_gt1_authorized") is not False:
        raise ValueError("legacy report cannot authorize batch>1")
    return report
