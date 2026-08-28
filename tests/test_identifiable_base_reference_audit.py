from __future__ import annotations

import pytest

from eg_sft.experiment.cpu_identifiability_audit import recompute_frozen_row
from eg_sft.experiment.identifiable_base_reference import audit_gsm_rows


def _row(record_id: str) -> dict:
    row = {"record_id": record_id, "raw_output": "Final answer: 7", "gold_value": "7"}
    row.update(recompute_frozen_row(row))
    return row


def test_gsm_base_reference_audit_recomputes_parser_and_order() -> None:
    report = audit_gsm_rows(
        rows=[_row("a"), _row("b")],
        frozen_records=[{"record_id": "a"}, {"record_id": "b"}],
    )
    assert report["status"] == "PASS"
    assert report["parser_rows_recomputed_from_raw_text"] == 2


def test_gsm_base_reference_audit_fails_on_parser_drift() -> None:
    row = _row("a")
    row["numeric_correct"] = False
    with pytest.raises(ValueError, match="parser recomputation"):
        audit_gsm_rows(rows=[row], frozen_records=[{"record_id": "a"}])
