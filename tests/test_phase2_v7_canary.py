from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eg_sft.evaluation.phase2_v7_canary import (
    CANARY_LEVELS,
    SEMANTIC_CANARY_LEVELS,
    canonical_json_bytes,
    compare_canary_signatures,
    select_canary_signatures,
    select_semantic_canary_signatures,
    validate_legacy_backend_report,
)


EOS = 151643


def _source_row(index: int) -> dict:
    return {
        "record_id": f"row-{index:02d}",
        "source_index": index,
        "question_sha256": f"{index:064x}",
        "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
        "generated_token_ids": [100 + index, EOS],
        "raw_output": f"Final answer: {index}",
        "parsed_prediction": str(index),
        "numeric_correct": True,
        "strict_parse_status": "ok",
        "parse_mode": "strict_final_marker",
        "parse_status": "ok",
        "gold_value": str(index),
    }


def test_canary_signatures_require_ordered_exact_16() -> None:
    source = [_source_row(index) for index in range(20)]
    ids = [row["record_id"] for row in source[:16]]
    signatures = select_canary_signatures(
        source_rows=source,
        selected_record_ids=ids,
        eos_token_id=EOS,
    )
    assert len(signatures) == 16
    assert signatures[0]["first_eos_ids"] == [100, EOS]
    assert compare_canary_signatures(
        reference=signatures, candidate=copy.deepcopy(signatures)
    )["status"] == "PASS"


def test_canary_detects_each_required_level() -> None:
    source = [_source_row(index) for index in range(16)]
    reference = select_canary_signatures(
        source_rows=source,
        selected_record_ids=[row["record_id"] for row in source],
        eos_token_id=EOS,
    )
    for level in CANARY_LEVELS:
        candidate = copy.deepcopy(reference)
        if level in {"raw_ids", "first_eos_ids"}:
            candidate[0][level] = [999, EOS]
        elif level == "correctness":
            candidate[0][level] = False
        else:
            candidate[0][level] = "changed"
        report = compare_canary_signatures(
            reference=reference, candidate=candidate
        )
        assert report["status"] == "FAIL"
        assert report["mismatch_counts"][level] == 1


def test_historical_adapter_semantic_reference_needs_no_token_ids() -> None:
    source = [_source_row(index) for index in range(16)]
    for row in source:
        row.pop("generated_token_ids")
    reference = select_semantic_canary_signatures(
        source_rows=source,
        selected_record_ids=[row["record_id"] for row in source],
    )
    candidate_source = [_source_row(index) for index in range(16)]
    candidate = select_canary_signatures(
        source_rows=candidate_source,
        selected_record_ids=[row["record_id"] for row in candidate_source],
        eos_token_id=EOS,
    )
    report = compare_canary_signatures(
        reference=reference,
        candidate=candidate,
        comparison_levels=SEMANTIC_CANARY_LEVELS,
    )
    assert report["status"] == "PASS"


def test_legacy_report_fails_closed_on_batch_gt_one(tmp_path: Path) -> None:
    environment_sha = "a" * 64
    audit = {
        "schema_version": "phase2-v7-canary-audit-v1",
        "status": "PASS",
        "role": "base_model_16",
        "record_count": 16,
        "exact_all_levels": True,
        "environment_contract_sha256": environment_sha,
    }
    report = {
        "schema_version": "phase2-v7-legacy-backend-validation-v1",
        "status": "LEGACY_BATCH1_VALIDATED",
        "environment_contract_sha256": environment_sha,
        "eval_backend": {
            "batch_size": 2,
            "padding_policy": "natural_per_example",
            "do_sample": False,
            "num_beams": 1,
            "max_input_tokens": 512,
            "max_new_tokens": 256,
            "dtype": "bf16",
            "attention_backend": "sdpa",
            "batch_gt1_authorized": False,
        },
        "batch_gt1_authorized": False,
        "canaries": {
            "base_model_16": audit,
            "archived_adapter_16": dict(audit, role="archived_adapter_16"),
        },
    }
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(ValueError, match="batch_size"):
        validate_legacy_backend_report(report_path=path)


def test_legacy_report_accepts_two_exact_canaries(tmp_path: Path) -> None:
    environment_sha = "b" * 64
    def audit(role: str) -> dict:
        return {
            "schema_version": "phase2-v7-canary-audit-v1",
            "status": "PASS",
            "role": role,
            "record_count": 16,
            "exact_all_levels": True,
            "environment_contract_sha256": environment_sha,
        }
    report = {
        "schema_version": "phase2-v7-legacy-backend-validation-v1",
        "status": "LEGACY_BATCH1_VALIDATED",
        "environment_contract_sha256": environment_sha,
        "eval_backend": {
            "batch_size": 1,
            "padding_policy": "natural_per_example",
            "do_sample": False,
            "num_beams": 1,
            "max_input_tokens": 512,
            "max_new_tokens": 256,
            "dtype": "bf16",
            "attention_backend": "sdpa",
            "batch_gt1_authorized": False,
        },
        "batch_gt1_authorized": False,
        "canaries": {
            "base_model_16": audit("base_model_16"),
            "archived_adapter_16": audit("archived_adapter_16"),
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_legacy_backend_report(report_path=path)["status"] == (
        "LEGACY_BATCH1_VALIDATED"
    )
