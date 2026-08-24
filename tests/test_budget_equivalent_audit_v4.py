from __future__ import annotations

from pathlib import Path

import pytest

from eg_sft.experiment.budget_equivalent_audit_v4 import (
    recompute_generation_row,
    validate_blind_merged_metrics,
)


def _row() -> dict[str, object]:
    return {
        "record_id": "gsm8k-test-0001",
        "raw_output": "Reasoning.\nFinal answer: 12",
        "gold_value": "12",
        "strict_parse_status": "ok",
        "strict_parsed_prediction": "12",
        "parse_mode": "strict_final_marker",
        "parse_status": "ok",
        "parsed_prediction": "12",
        "numeric_correct": True,
    }


def test_recompute_generation_row_accepts_exact_parser_evidence() -> None:
    assert recompute_generation_row(_row())["numeric_correct"] is True


def test_recompute_generation_row_rejects_mutated_correctness() -> None:
    row = _row()
    row["numeric_correct"] = False
    with pytest.raises(ValueError, match="stored parser field changed"):
        recompute_generation_row(row)


def test_validate_blind_metrics_accepts_metadata_only(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    from eg_sft.training.b500 import file_sha256

    validate_blind_merged_metrics(
        metrics={
            "status": "PASS",
            "accuracy_withheld": True,
            "record_count": 1,
            "raw_outputs_sha256": file_sha256(raw_path),
            "worker_count": 2,
            "workers": [{"record_count": 659}, {"record_count": 660}],
        },
        rows=[{}],
        raw_path=raw_path,
    )


def test_validate_blind_metrics_rejects_missing_record_count(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record count"):
        validate_blind_merged_metrics(
            metrics={
                "status": "PASS",
                "accuracy_withheld": True,
                "raw_outputs_sha256": "x",
                "worker_count": 2,
                "workers": [{"record_count": 659}, {"record_count": 660}],
            },
            rows=[{}],
            raw_path=raw_path,
        )
