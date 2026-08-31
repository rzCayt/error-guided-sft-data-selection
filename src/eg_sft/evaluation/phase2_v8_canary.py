"""Pure signature, comparison and backend-report controls for v8 canaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eg_sft.evaluation.phase2_v7_canary import file_sha256, read_json


FULL_LEVELS = (
    "prompt_token_ids",
    "attention_mask",
    "raw_continuation_ids",
    "first_eos_continuation_ids",
    "decoded_canonical_text",
    "parser_input_text",
    "parsed_number",
    "correctness",
    "strict_status",
)
HISTORICAL_BASE_LEVELS = (
    "raw_continuation_ids",
    "first_eos_continuation_ids",
    "decoded_canonical_text",
    "parsed_number",
    "correctness",
    "strict_status",
)
HISTORICAL_ADAPTER_LEVELS = (
    "decoded_canonical_text",
    "parser_input_text",
    "parsed_number",
    "correctness",
    "strict_status",
)


def canonical_first_eos(token_ids: Sequence[int], eos_ids: Sequence[int]) -> list[int]:
    values = [int(value) for value in token_ids]
    eos = {int(value) for value in eos_ids}
    end = next((index + 1 for index, value in enumerate(values) if value in eos), len(values))
    return values[:end]


def v8_signature(
    row: Mapping[str, Any], *, prompt_ids: Sequence[int], attention_mask: Sequence[int]
) -> dict[str, Any]:
    return {
        "record_id": str(row["record_id"]),
        "source_index": int(row["source_index"]),
        "question_sha256": str(row["question_sha256"]),
        "prompt_version": str(row["prompt_version"]),
        "prompt_token_ids": [int(value) for value in prompt_ids],
        "attention_mask": [int(value) for value in attention_mask],
        "raw_continuation_ids": [int(value) for value in row["raw_continuation_ids"]],
        "first_eos_continuation_ids": [
            int(value) for value in row["first_eos_continuation_ids"]
        ],
        "decoded_canonical_text": str(row["raw_output"]),
        "parser_input_text": str(row["parser_input_text"]),
        "parsed_number": row.get("parsed_prediction"),
        "correctness": bool(row.get("numeric_correct")),
        "strict_status": str(row["strict_parse_status"]),
        "parse_mode": str(row["parse_mode"]),
        "parse_status": str(row["parse_status"]),
        "gold_value": str(row["gold_value"]),
    }


def compare_v8_signatures(
    *,
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    levels: Sequence[str],
    expected_count: int,
) -> dict[str, Any]:
    if len(reference) != expected_count or len(candidate) != expected_count:
        raise ValueError("v8 canary record count changed")
    reference_ids = [str(row["record_id"]) for row in reference]
    candidate_ids = [str(row["record_id"]) for row in candidate]
    if reference_ids != candidate_ids or len(set(reference_ids)) != expected_count:
        raise ValueError("v8 canary record IDs/order changed")
    unknown = [level for level in levels if level not in FULL_LEVELS]
    if not levels or unknown:
        raise ValueError(f"unknown v8 comparison levels: {unknown}")
    mismatch = {str(level): [] for level in levels}
    for left, right in zip(reference, candidate, strict=True):
        for level in levels:
            if left.get(level) != right.get(level):
                mismatch[str(level)].append(str(left["record_id"]))
        for identity in (
            "source_index",
            "question_sha256",
            "prompt_version",
            "gold_value",
        ):
            if left.get(identity) != right.get(identity):
                raise ValueError(f"v8 canary identity changed: {identity}")
    counts = {key: len(value) for key, value in mismatch.items()}
    passed = all(value == 0 for value in counts.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "record_count": expected_count,
        "comparison_levels": list(levels),
        "mismatch_counts": counts,
        "mismatch_record_ids": mismatch,
        "exact_all_levels": passed,
    }


def validate_v8_backend_report(
    *,
    report_path: Path,
    expected_sha256: str | None = None,
    expected_worker_id: str | None = None,
    expected_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    path = report_path.resolve()
    if not path.is_file():
        raise ValueError("v8 backend report does not exist")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("v8 backend report SHA-256 changed")
    report = read_json(path)
    if report.get("schema_version") != "phase2-v8-legacy-backend-validation-v1":
        raise ValueError("unexpected v8 backend report schema")
    if report.get("status") != "LEGACY_BATCH1_VALIDATED":
        raise ValueError("v8 legacy batch1 backend has not been validated")
    backend = report.get("eval_backend", {})
    required = {
        "batch_size": 1,
        "padding_policy": "natural_per_example",
        "do_sample": False,
        "num_beams": 1,
        "max_input_tokens": 512,
        "max_new_tokens": 256,
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "batch_gt1_authorized": False,
    }
    for field, expected in required.items():
        if backend.get(field) != expected:
            raise ValueError(f"v8 backend field changed: {field}")
    if len(str(report.get("environment_contract_sha256", ""))) != 64:
        raise ValueError("v8 backend environment SHA is invalid")
    canary_contract_sha = str(report.get("canary_contract_sha256", ""))
    if len(canary_contract_sha) != 64 or any(
        character not in "0123456789abcdef" for character in canary_contract_sha
    ):
        raise ValueError("v8 backend canary-contract SHA is invalid")
    if report.get("base_new_block_exact") is not True:
        raise ValueError("v8 base new-block exact canary did not pass")
    if report.get("adapter_historical_semantic_bridge") is not True:
        raise ValueError("v8 historical adapter semantic bridge did not pass")
    if report.get("adapter_new_block_token_exact") is not True:
        raise ValueError("v8 adapter new-block token anchor did not pass")
    if report.get("batch_gt1_authorized") is not False:
        raise ValueError("v8 backend report cannot authorize batch>1")
    if expected_worker_id is not None and report.get("worker_id") != expected_worker_id:
        raise ValueError("v8 backend worker identity changed")
    if expected_gpu_uuid is not None and report.get("gpu_uuid") != expected_gpu_uuid:
        raise ValueError("v8 backend GPU UUID changed")
    return report
