from __future__ import annotations

import pytest

from eg_sft.experiment.budget_equivalent_ood_audit_v2 import (
    audit_complete_ood_dataset_from_raw,
    recompute_ood_generation_row,
)


def _frozen() -> dict[str, object]:
    from eg_sft.data.public_gsm8k import sha256_text

    return {
        "record_id": "svamp-00000-hash",
        "dataset": "svamp",
        "source_index": 0,
        "question_sha256": "q",
        "gold_value_sha256": sha256_text("5"),
    }


def _row() -> dict[str, object]:
    return {
        "record_id": "svamp-00000-hash",
        "dataset": "svamp",
        "source_index": 0,
        "question_sha256": "q",
        "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
        "raw_output": "Calculation.\nFinal answer: 5",
        "gold_value": "5",
        "strict_parse_status": "ok",
        "parse_mode": "strict_final_marker",
        "parse_status": "ok",
        "parsed_prediction": "5",
        "numeric_correct": True,
    }


def test_recompute_ood_row_accepts_exact_evidence() -> None:
    assert recompute_ood_generation_row(_row())["numeric_correct"] is True


def test_recompute_ood_row_rejects_mutation() -> None:
    row = _row()
    row["parsed_prediction"] = "4"
    with pytest.raises(ValueError, match="stored OOD parser field changed"):
        recompute_ood_generation_row(row)


def test_complete_ood_audit_recomputes_raw_parser() -> None:
    report = audit_complete_ood_dataset_from_raw(
        rows=[_row()],
        frozen_records=[_frozen()],
    )
    assert report["status"] == "PASS"
    assert report["parser_rows_recomputed_from_raw_text"] == 1
