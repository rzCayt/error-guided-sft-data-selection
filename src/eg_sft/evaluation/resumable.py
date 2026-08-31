"""Integrity checks and metric aggregation for resumable GSM8K evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def validate_completed_prefix(
    *,
    completed_rows: Sequence[dict[str, Any]],
    frozen_records: Sequence[dict[str, Any]],
) -> int:
    """Require completed outputs to be an exact prefix of the frozen test set."""

    if len(completed_rows) > len(frozen_records):
        raise ValueError("completed outputs exceed the frozen test set")
    seen: set[str] = set()
    for index, row in enumerate(completed_rows):
        record_id = str(row.get("record_id", ""))
        if not record_id:
            raise ValueError(f"completed row {index} has no record_id")
        if record_id in seen:
            raise ValueError(f"duplicate completed record_id: {record_id}")
        seen.add(record_id)
        expected_id = str(frozen_records[index]["record_id"])
        if record_id != expected_id:
            raise ValueError(
                f"completed outputs are not a frozen prefix at index {index}: "
                f"{record_id} != {expected_id}"
            )
    return len(completed_rows)


def aggregate_gsm8k_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation")
    correct = sum(bool(row["numeric_correct"]) for row in rows)
    parsed = sum(row["parse_status"] == "ok" for row in rows)
    strict = sum(row["strict_parse_status"] == "ok" for row in rows)
    fallback = sum(
        row["parse_mode"] == "last_numeric_fallback" for row in rows
    )
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["parse_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "example_count": len(rows),
        "numeric_correct_count": correct,
        "numeric_accuracy": correct / len(rows),
        "parsed_count": parsed,
        "parse_rate": parsed / len(rows),
        "strict_parsed_count": strict,
        "strict_parse_rate": strict / len(rows),
        "fallback_parsed_count": fallback,
        "parse_status_counts": status_counts,
    }
