"""CPU-only integrity checks for the four-task base-model reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eg_sft.experiment.cpu_identifiability_audit import parser_mismatches


def audit_gsm_rows(
    *, rows: Sequence[Mapping[str, Any]], frozen_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(rows) != len(frozen_records):
        raise ValueError("GSM8K base-reference output count changed")
    ids: list[str] = []
    mismatch_count = 0
    for index, (row, frozen) in enumerate(zip(rows, frozen_records, strict=True)):
        if row.get("record_id") != frozen.get("record_id"):
            raise ValueError(f"GSM8K base-reference order changed at row {index}")
        ids.append(str(row["record_id"]))
        mismatches = parser_mismatches(dict(row))
        if mismatches:
            mismatch_count += 1
    if len(ids) != len(set(ids)):
        raise ValueError("GSM8K base-reference contains duplicate record IDs")
    if mismatch_count:
        raise ValueError("GSM8K base-reference parser recomputation changed")
    return {
        "status": "PASS",
        "record_count": len(rows),
        "unique_record_id_count": len(ids),
        "ordered_frozen_membership": True,
        "parser_rows_recomputed_from_raw_text": len(rows),
        "parser_mismatch_count": 0,
        "accuracy_withheld": True,
    }
