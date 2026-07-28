import pytest

from eg_sft.evaluation.resumable import (
    aggregate_gsm8k_metrics,
    validate_completed_prefix,
)


def _records() -> list[dict]:
    return [{"record_id": f"r{index}"} for index in range(3)]


def test_completed_outputs_must_be_an_exact_frozen_prefix() -> None:
    assert validate_completed_prefix(
        completed_rows=[{"record_id": "r0"}, {"record_id": "r1"}],
        frozen_records=_records(),
    ) == 2


def test_completed_outputs_reject_gap_or_reordering() -> None:
    with pytest.raises(ValueError, match="not a frozen prefix"):
        validate_completed_prefix(
            completed_rows=[{"record_id": "r0"}, {"record_id": "r2"}],
            frozen_records=_records(),
        )


def test_completed_outputs_reject_duplicate_id() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_completed_prefix(
            completed_rows=[{"record_id": "r0"}, {"record_id": "r0"}],
            frozen_records=_records(),
        )


def test_metric_aggregation_keeps_format_and_numeric_accuracy_separate() -> None:
    rows = [
        {
            "numeric_correct": True,
            "parse_status": "ok",
            "strict_parse_status": "ok",
            "parse_mode": "strict_final_marker",
        },
        {
            "numeric_correct": False,
            "parse_status": "ok",
            "strict_parse_status": "missing_final_marker",
            "parse_mode": "last_numeric_fallback",
        },
    ]
    metrics = aggregate_gsm8k_metrics(rows)
    assert metrics["numeric_accuracy"] == 0.5
    assert metrics["parse_rate"] == 1.0
    assert metrics["strict_parse_rate"] == 0.5
    assert metrics["fallback_parsed_count"] == 1
