"""Accuracy-blind qualification helpers for a batched Transformers backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


IDENTIFIABLE_MATRIX_VERSION = "identifiable-budget-v4-extension-v1"
PHASE2_MATRIX_VERSION = "phase2-crossed-48cell-v7"
PHASE2_CLEAN_COMMON_V8 = "phase2-clean-common24-v8"
ALLOWED_EVAL_BATCH_SIZES = (1, 2, 4, 8)
ROW_EQUIVALENCE_FIELDS = (
    "record_id",
    "parsed_prediction",
    "numeric_correct",
    "strict_parse_status",
    "parse_mode",
    "parse_status",
)
QUALIFICATION_GATES = (
    "row_level_equivalence",
    "token_ids_equal_or_full_shadow_semantic_equivalence",
    "throughput_at_least_1_5x",
    "full_cell_wall_time_reduction_at_least_25_percent",
    "resume_without_gap_or_duplicate",
    "output_non_overwrite",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_qualification_artifact(
    *, report_path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Bind batched execution to one immutable, fully passed qualification."""

    report_path = report_path.resolve()
    if not report_path.is_file():
        raise ValueError("qualification report does not exist")
    actual_sha256 = _file_sha256(report_path)
    if actual_sha256 != expected_sha256.strip().lower():
        raise ValueError("qualification report SHA-256 changed")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        raise ValueError("qualification report has not passed")
    gates = payload.get("gates", {})
    if any(gates.get(name) is not True for name in QUALIFICATION_GATES):
        raise ValueError("qualification report gates are incomplete")
    return payload


def resolve_eval_batch_size(
    *,
    matrix_config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> tuple[int, bool]:
    """Resolve the opt-in physical batch size without weakening v3 contracts."""

    raw = environ.get("EG_SFT_EVAL_BATCH_SIZE", "1").strip()
    try:
        batch_size = int(raw)
    except ValueError as error:
        raise ValueError("EG_SFT_EVAL_BATCH_SIZE must be an integer") from error
    if batch_size not in ALLOWED_EVAL_BATCH_SIZES:
        raise ValueError("EG_SFT_EVAL_BATCH_SIZE must be one of 1, 2, 4, 8")

    matrix_version = matrix_config.get("matrix_version")
    identifiable_v4 = matrix_version in {
        IDENTIFIABLE_MATRIX_VERSION,
        PHASE2_MATRIX_VERSION,
        PHASE2_CLEAN_COMMON_V8,
    }
    if matrix_version in {PHASE2_MATRIX_VERSION, PHASE2_CLEAN_COMMON_V8}:
        if batch_size != 1:
            raise ValueError("Phase-2 legacy evaluation requires batch size 1")
        report = environ.get("EG_SFT_LEGACY_BACKEND_REPORT", "").strip()
        report_sha256 = environ.get(
            "EG_SFT_LEGACY_BACKEND_REPORT_SHA256", ""
        ).strip()
        environment_sha = environ.get(
            "EG_SFT_ENVIRONMENT_CONTRACT_SHA256", ""
        ).strip()
        if not report or not report_sha256 or not environment_sha:
            raise ValueError(
                "Phase-2 requires a bound legacy backend report and environment SHA"
            )
        if matrix_version == PHASE2_CLEAN_COMMON_V8:
            from eg_sft.evaluation.phase2_v8_canary import (
                validate_v8_backend_report,
            )

            payload = validate_v8_backend_report(
                report_path=Path(report), expected_sha256=report_sha256
            )
        else:
            from eg_sft.evaluation.phase2_v7_canary import (
                validate_legacy_backend_report,
            )

            payload = validate_legacy_backend_report(
                report_path=Path(report),
                expected_sha256=report_sha256,
            )
        if payload.get("environment_contract_sha256") != environment_sha:
            raise ValueError("Phase-2 legacy report uses another environment")
    if batch_size > 1:
        if not identifiable_v4:
            raise ValueError("batched evaluation is forbidden for legacy matrices")
        if environ.get("EG_SFT_WORKER_ID", "").strip() not in {"gpu0", "gpu1"}:
            raise ValueError(
                "batched evaluation requires the qualified identifiable-v4 controller"
            )
        report = environ.get("EG_SFT_QUALIFICATION_REPORT", "").strip()
        report_sha256 = environ.get(
            "EG_SFT_QUALIFICATION_REPORT_SHA256", ""
        ).strip()
        if not report or not report_sha256:
            raise ValueError(
                "batched evaluation requires a bound qualification report and SHA-256"
            )
        validate_qualification_artifact(
            report_path=Path(report),
            expected_sha256=report_sha256,
        )
    return batch_size, identifiable_v4


def generated_token_rows(
    *, generated_ids: Any, padded_input_width: int
) -> list[list[int]]:
    """Extract only generated token IDs from a left-padded batch.

    ``transformers.generate`` returns the padded prompt followed by generated
    tokens.  All rows therefore share ``padded_input_width`` even when their
    unpadded prompt lengths differ.
    """

    if padded_input_width < 0:
        raise ValueError("padded_input_width must be non-negative")
    rows = generated_ids.tolist() if hasattr(generated_ids, "tolist") else generated_ids
    result = [list(map(int, row[padded_input_width:])) for row in rows]
    if not result:
        raise ValueError("generated batch is empty")
    return result


def record_generated_token_ids(
    *,
    scored_row: dict[str, Any],
    token_ids: Sequence[int],
    identifiable_v4: bool,
    eos_token_id: int | None = None,
    canonical_decoded_text: str | None = None,
    parser_input: str | None = None,
) -> None:
    """Attach full and first-EOS token evidence only to new v4 outputs."""

    if identifiable_v4:
        full = [int(value) for value in token_ids]
        scored_row["generated_token_ids"] = full
        if eos_token_id is not None:
            try:
                end = full.index(int(eos_token_id)) + 1
            except ValueError:
                end = len(full)
            scored_row["first_eos_generated_token_ids"] = full[:end]
            scored_row["first_eos_index"] = end - 1 if end < len(full) or (
                full and full[-1] == int(eos_token_id)
            ) else None
        if canonical_decoded_text is not None:
            scored_row["canonical_decoded_text"] = str(canonical_decoded_text)
        if parser_input is not None:
            scored_row["parser_input"] = str(parser_input)


def validate_phase2_generation_evidence(
    rows: Sequence[Mapping[str, Any]], *, eos_token_id: int
) -> None:
    """Require auditable raw/first-EOS/parser evidence for every Phase-2 row."""

    if not rows:
        raise ValueError("Phase-2 generation evidence is empty")
    for row in rows:
        record_id = str(row.get("record_id", "UNKNOWN"))
        full = row.get("generated_token_ids")
        canonical = row.get("first_eos_generated_token_ids")
        if not isinstance(full, list) or not isinstance(canonical, list):
            raise ValueError(f"Phase-2 token evidence is missing: {record_id}")
        full_ids = [int(value) for value in full]
        try:
            end = full_ids.index(int(eos_token_id)) + 1
        except ValueError:
            end = len(full_ids)
        if [int(value) for value in canonical] != full_ids[:end]:
            raise ValueError(f"Phase-2 first-EOS evidence changed: {record_id}")
        expected_index = (
            end - 1
            if full_ids and end > 0 and full_ids[end - 1] == int(eos_token_id)
            else None
        )
        if row.get("first_eos_index") != expected_index:
            raise ValueError(f"Phase-2 first-EOS index changed: {record_id}")
        raw = str(row.get("raw_output", ""))
        if row.get("canonical_decoded_text") != raw or row.get("parser_input") != raw:
            raise ValueError(f"Phase-2 decoded/parser evidence changed: {record_id}")


def validate_resumable_batch_prefix(
    *, completed_rows: Sequence[Mapping[str, Any]], frozen_ids: Sequence[str]
) -> int:
    """Require an ordered prefix so a crashed batch can be safely retried."""

    if len(completed_rows) > len(frozen_ids):
        raise ValueError("completed output is longer than the frozen records")
    observed = [str(row.get("record_id")) for row in completed_rows]
    expected = [str(value) for value in frozen_ids[: len(observed)]]
    if observed != expected:
        raise ValueError("completed output is not an ordered frozen prefix")
    if len(observed) != len(set(observed)):
        raise ValueError("completed output contains duplicate record IDs")
    return len(observed)


def compare_backend_rows(
    *,
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare token IDs when available and always compare scored row semantics."""

    if len(reference) != len(candidate):
        return {
            "status": "FAIL",
            "reason": "record_count_mismatch",
            "reference_count": len(reference),
            "candidate_count": len(candidate),
        }
    semantic_mismatches: list[dict[str, Any]] = []
    token_mismatches: list[str] = []
    token_comparable = True
    for left, right in zip(reference, candidate, strict=True):
        record_id = str(left.get("record_id"))
        differences = {}
        for field in ROW_EQUIVALENCE_FIELDS:
            left_present = field in left
            right_present = field in right
            if not left_present or not right_present:
                differences[field] = {
                    "reference": left.get(field),
                    "candidate": right.get(field),
                    "reference_present": left_present,
                    "candidate_present": right_present,
                }
            elif left[field] != right[field]:
                differences[field] = {
                    "reference": left[field],
                    "candidate": right[field],
                }
        if differences:
            semantic_mismatches.append(
                {"record_id": record_id, "differences": differences}
            )
        if "generated_token_ids" not in left or "generated_token_ids" not in right:
            token_comparable = False
        elif list(left["generated_token_ids"]) != list(right["generated_token_ids"]):
            token_mismatches.append(record_id)
    semantic_equal = not semantic_mismatches
    token_equal = token_comparable and not token_mismatches
    return {
        "status": "PASS" if semantic_equal else "FAIL",
        "record_count": len(reference),
        "semantic_equal": semantic_equal,
        "token_ids_comparable": token_comparable,
        "token_ids_equal": token_equal,
        "semantic_mismatch_count": len(semantic_mismatches),
        "token_mismatch_count": len(token_mismatches),
        "semantic_mismatches": semantic_mismatches[:20],
        "token_mismatch_record_ids": token_mismatches[:20],
    }


def qualification_decision(
    *,
    row_comparison: Mapping[str, Any],
    reference_examples_per_second: float,
    candidate_examples_per_second: float,
    reference_full_cell_seconds: float,
    candidate_full_cell_seconds: float,
    resume_passed: bool,
    non_overwrite_passed: bool,
) -> dict[str, Any]:
    """Apply the frozen 1.5x throughput and 25% wall-time gates."""

    positive = (
        reference_examples_per_second > 0
        and candidate_examples_per_second > 0
        and reference_full_cell_seconds > 0
        and candidate_full_cell_seconds > 0
    )
    if not positive:
        raise ValueError("qualification timing values must be positive")
    speedup = candidate_examples_per_second / reference_examples_per_second
    wall_reduction = 1.0 - candidate_full_cell_seconds / reference_full_cell_seconds
    semantic_pass = row_comparison.get("status") == "PASS"
    full_shadow_semantic_equivalence = (
        semantic_pass and int(row_comparison.get("record_count", -1)) == 3841
    )
    token_or_full_shadow_pass = (
        row_comparison.get("token_ids_equal") is True
        or full_shadow_semantic_equivalence
    )
    gates = {
        "row_level_equivalence": semantic_pass,
        "token_ids_equal_or_full_shadow_semantic_equivalence": (
            token_or_full_shadow_pass
        ),
        "throughput_at_least_1_5x": speedup >= 1.5,
        "full_cell_wall_time_reduction_at_least_25_percent": wall_reduction >= 0.25,
        "resume_without_gap_or_duplicate": bool(resume_passed),
        "output_non_overwrite": bool(non_overwrite_passed),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "throughput_speedup": speedup,
        "full_cell_wall_time_reduction": wall_reduction,
        "token_id_equivalence": row_comparison.get("token_ids_equal"),
        "full_shadow_semantic_equivalence": full_shadow_semantic_equivalence,
        "token_id_policy": (
            "Token IDs are preferred. A mismatch or unavailable comparison is "
            "accepted only when all 3,841 full-shadow rows are semantically equal."
        ),
        "fallback_backend": "batch1_transformers" if not all(gates.values()) else None,
    }
