from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eg_sft.evaluation.identifiable_batch_backend import (
    QUALIFICATION_GATES,
    compare_backend_rows,
    generated_token_rows,
    qualification_decision,
    record_generated_token_ids,
    resolve_eval_batch_size,
    validate_resumable_batch_prefix,
    validate_phase2_generation_evidence,
)


def _passed_qualification(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gates": {name: True for name in QUALIFICATION_GATES},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _passed_legacy_report(tmp_path: Path) -> tuple[Path, str, str]:
    environment_sha = "e" * 64
    def audit(role: str) -> dict:
        return {
            "schema_version": "phase2-v7-canary-audit-v1",
            "status": "PASS",
            "role": role,
            "record_count": 16,
            "exact_all_levels": True,
            "environment_contract_sha256": environment_sha,
        }
    payload = {
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
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), environment_sha


def test_eval_batch_size_defaults_to_one_and_rejects_legacy_batching() -> None:
    assert resolve_eval_batch_size(matrix_config={}, environ={}) == (1, False)
    with pytest.raises(ValueError, match="forbidden for legacy"):
        resolve_eval_batch_size(
            matrix_config={},
            environ={"EG_SFT_EVAL_BATCH_SIZE": "2"},
        )


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_eval_batch_size_requires_v4_qualified_controller(
    batch_size: int, tmp_path: Path
) -> None:
    matrix = {"matrix_version": "identifiable-budget-v4-extension-v1"}
    with pytest.raises(ValueError, match="qualified identifiable-v4 controller"):
        resolve_eval_batch_size(
            matrix_config=matrix,
            environ={"EG_SFT_EVAL_BATCH_SIZE": str(batch_size)},
        )
    with pytest.raises(ValueError, match="bound qualification report"):
        resolve_eval_batch_size(
            matrix_config=matrix,
            environ={
                "EG_SFT_EVAL_BATCH_SIZE": str(batch_size),
                "EG_SFT_WORKER_ID": "gpu0",
            },
        )
    report, report_sha256 = _passed_qualification(tmp_path)
    assert resolve_eval_batch_size(
        matrix_config=matrix,
        environ={
            "EG_SFT_EVAL_BATCH_SIZE": str(batch_size),
            "EG_SFT_WORKER_ID": "gpu0",
            "EG_SFT_QUALIFICATION_REPORT": str(report),
            "EG_SFT_QUALIFICATION_REPORT_SHA256": report_sha256,
        },
    ) == (batch_size, True)


def test_eval_batch_size_rejects_changed_qualification_artifact(
    tmp_path: Path,
) -> None:
    report, report_sha256 = _passed_qualification(tmp_path)
    report.write_text('{"status":"FAIL","gates":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        resolve_eval_batch_size(
            matrix_config={"matrix_version": "identifiable-budget-v4-extension-v1"},
            environ={
                "EG_SFT_EVAL_BATCH_SIZE": "4",
                "EG_SFT_WORKER_ID": "gpu0",
                "EG_SFT_QUALIFICATION_REPORT": str(report),
                "EG_SFT_QUALIFICATION_REPORT_SHA256": report_sha256,
            },
        )


def test_phase2_requires_bound_legacy_batch1_report(tmp_path: Path) -> None:
    matrix = {"matrix_version": "phase2-crossed-48cell-v7"}
    with pytest.raises(ValueError, match="bound legacy"):
        resolve_eval_batch_size(matrix_config=matrix, environ={})
    report, report_sha, environment_sha = _passed_legacy_report(tmp_path)
    assert resolve_eval_batch_size(
        matrix_config=matrix,
        environ={
            "EG_SFT_EVAL_BATCH_SIZE": "1",
            "EG_SFT_LEGACY_BACKEND_REPORT": str(report),
            "EG_SFT_LEGACY_BACKEND_REPORT_SHA256": report_sha,
            "EG_SFT_ENVIRONMENT_CONTRACT_SHA256": environment_sha,
        },
    ) == (1, True)
    with pytest.raises(ValueError, match="batch size 1"):
        resolve_eval_batch_size(
            matrix_config=matrix,
            environ={
                "EG_SFT_EVAL_BATCH_SIZE": "2",
                "EG_SFT_LEGACY_BACKEND_REPORT": str(report),
                "EG_SFT_LEGACY_BACKEND_REPORT_SHA256": report_sha,
                "EG_SFT_ENVIRONMENT_CONTRACT_SHA256": environment_sha,
            },
        )


@pytest.mark.parametrize("raw", ["0", "3", "16", "not-an-int"])
def test_eval_batch_size_rejects_unsupported_values(raw: str) -> None:
    with pytest.raises(ValueError, match="EG_SFT_EVAL_BATCH_SIZE"):
        resolve_eval_batch_size(
            matrix_config={"matrix_version": "identifiable-budget-v4-extension-v1"},
            environ={"EG_SFT_EVAL_BATCH_SIZE": raw, "EG_SFT_WORKER_ID": "gpu0"},
        )


def test_generated_token_rows_slice_the_shared_padded_width() -> None:
    assert generated_token_rows(
        generated_ids=[[0, 0, 11, 12, 90, 91], [0, 21, 22, 23, 80, 81]],
        padded_input_width=4,
    ) == [[90, 91], [80, 81]]


def test_generated_token_ids_are_v4_only() -> None:
    legacy = {"record_id": "legacy"}
    record_generated_token_ids(
        scored_row=legacy, token_ids=[90, 91], identifiable_v4=False
    )
    assert "generated_token_ids" not in legacy

    v4 = {"record_id": "v4"}
    record_generated_token_ids(
        scored_row=v4,
        token_ids=[90, 151643, 91],
        identifiable_v4=True,
        eos_token_id=151643,
        canonical_decoded_text="answer",
        parser_input="answer",
    )
    assert v4["generated_token_ids"] == [90, 151643, 91]
    assert v4["first_eos_generated_token_ids"] == [90, 151643]
    assert v4["first_eos_index"] == 1
    assert v4["canonical_decoded_text"] == "answer"
    assert v4["parser_input"] == "answer"


def test_phase2_generation_evidence_fails_closed() -> None:
    row = {
        "record_id": "a",
        "raw_output": "answer",
        "generated_token_ids": [90, 151643, 91],
        "first_eos_generated_token_ids": [90, 151643],
        "first_eos_index": 1,
        "canonical_decoded_text": "answer",
        "parser_input": "answer",
    }
    validate_phase2_generation_evidence([row], eos_token_id=151643)
    broken = dict(row, first_eos_generated_token_ids=[90, 151643, 91])
    with pytest.raises(ValueError, match="first-EOS"):
        validate_phase2_generation_evidence([broken], eos_token_id=151643)


def test_resume_requires_unique_ordered_prefix() -> None:
    assert validate_resumable_batch_prefix(
        completed_rows=[{"record_id": "a"}, {"record_id": "b"}],
        frozen_ids=["a", "b", "c"],
    ) == 2
    with pytest.raises(ValueError, match="ordered frozen prefix"):
        validate_resumable_batch_prefix(
            completed_rows=[{"record_id": "b"}], frozen_ids=["a", "b"]
        )


def test_backend_semantics_can_pass_when_token_ids_are_not_recorded() -> None:
    rows = [
        {
            "record_id": "a",
            "parsed_prediction": "7",
            "numeric_correct": True,
            "strict_parse_status": "ok",
            "parse_mode": "strict_final_marker",
            "parse_status": "ok",
        }
    ]
    report = compare_backend_rows(reference=rows, candidate=rows)
    assert report["status"] == "PASS"
    assert report["token_ids_comparable"] is False


def test_backend_comparison_fails_when_parsed_prediction_changes() -> None:
    reference = [
        {
            "record_id": "a",
            "parsed_prediction": "7",
            "numeric_correct": True,
            "strict_parse_status": "ok",
            "parse_mode": "strict_final_marker",
            "parse_status": "ok",
            "generated_token_ids": [90, 91],
        }
    ]
    candidate = [{**reference[0], "parsed_prediction": "8"}]
    report = compare_backend_rows(reference=reference, candidate=candidate)
    assert report["status"] == "FAIL"
    assert report["semantic_mismatches"][0]["differences"][
        "parsed_prediction"
    ] == {"reference": "7", "candidate": "8"}


def test_backend_comparison_rejects_missing_semantic_fields() -> None:
    incomplete = [{"record_id": "a", "generated_token_ids": [90]}]
    report = compare_backend_rows(reference=incomplete, candidate=incomplete)
    assert report["status"] == "FAIL"
    assert "parsed_prediction" in report["semantic_mismatches"][0]["differences"]
    assert report["semantic_mismatches"][0]["differences"]["parsed_prediction"][
        "reference_present"
    ] is False


def test_qualification_applies_all_frozen_gates() -> None:
    report = qualification_decision(
        row_comparison={"status": "PASS", "token_ids_equal": True},
        reference_examples_per_second=1.0,
        candidate_examples_per_second=1.6,
        reference_full_cell_seconds=100.0,
        candidate_full_cell_seconds=70.0,
        resume_passed=True,
        non_overwrite_passed=True,
    )
    assert report["status"] == "PASS"
    assert report["gates"][
        "token_ids_equal_or_full_shadow_semantic_equivalence"
    ] is True
    token_failed = qualification_decision(
        row_comparison={"status": "PASS", "token_ids_equal": False},
        reference_examples_per_second=1.0,
        candidate_examples_per_second=1.6,
        reference_full_cell_seconds=100.0,
        candidate_full_cell_seconds=70.0,
        resume_passed=True,
        non_overwrite_passed=True,
    )
    assert token_failed["status"] == "FAIL"
    assert token_failed["gates"][
        "token_ids_equal_or_full_shadow_semantic_equivalence"
    ] is False
    failed = qualification_decision(
        row_comparison={"status": "PASS", "token_ids_equal": True},
        reference_examples_per_second=1.0,
        candidate_examples_per_second=1.4,
        reference_full_cell_seconds=100.0,
        candidate_full_cell_seconds=70.0,
        resume_passed=True,
        non_overwrite_passed=True,
    )
    assert failed["status"] == "FAIL"
    assert failed["fallback_backend"] == "batch1_transformers"


def _qualified_row(index: int, *, token_id: int, prediction: str | None = None) -> dict:
    return {
        "record_id": f"r{index:04d}",
        "parsed_prediction": str(index) if prediction is None else prediction,
        "numeric_correct": True,
        "strict_parse_status": "ok",
        "parse_mode": "strict_final_marker",
        "parse_status": "ok",
        "generated_token_ids": [token_id],
    }


def _decision_from_rows(reference: list[dict], candidate: list[dict]) -> dict:
    return qualification_decision(
        row_comparison=compare_backend_rows(
            reference=reference,
            candidate=candidate,
        ),
        reference_examples_per_second=1.0,
        candidate_examples_per_second=1.6,
        reference_full_cell_seconds=100.0,
        candidate_full_cell_seconds=70.0,
        resume_passed=True,
        non_overwrite_passed=True,
    )


def test_128_semantic_equal_but_token_different_fails() -> None:
    reference = [_qualified_row(index, token_id=index) for index in range(128)]
    candidate = [_qualified_row(index, token_id=index + 1) for index in range(128)]
    decision = _decision_from_rows(reference, candidate)
    assert decision["status"] == "FAIL"
    assert decision["full_shadow_semantic_equivalence"] is False
    assert decision["gates"][
        "token_ids_equal_or_full_shadow_semantic_equivalence"
    ] is False


def test_3841_semantic_equal_but_token_different_passes() -> None:
    reference = [_qualified_row(index, token_id=index) for index in range(3841)]
    candidate = [_qualified_row(index, token_id=index + 1) for index in range(3841)]
    decision = _decision_from_rows(reference, candidate)
    assert decision["status"] == "PASS"
    assert decision["token_id_equivalence"] is False
    assert decision["full_shadow_semantic_equivalence"] is True
    assert decision["gates"][
        "token_ids_equal_or_full_shadow_semantic_equivalence"
    ] is True


def test_3841_one_semantic_mismatch_fails() -> None:
    reference = [_qualified_row(index, token_id=index) for index in range(3841)]
    candidate = [_qualified_row(index, token_id=index + 1) for index in range(3841)]
    candidate[2000]["parsed_prediction"] = "different"
    decision = _decision_from_rows(reference, candidate)
    assert decision["status"] == "FAIL"
    assert decision["full_shadow_semantic_equivalence"] is False
    assert decision["gates"]["row_level_equivalence"] is False
