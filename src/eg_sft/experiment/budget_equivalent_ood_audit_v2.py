"""Raw-text parser recomputation for arithmetic OOD audit artifacts."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from eg_sft.experiment.budget_equivalent_ood_runtime import audit_complete_dataset
from eg_sft.gsm8k.parser import parse_generated_answer, parse_last_numeric_answer


def recompute_ood_generation_row(row: dict[str, Any]) -> dict[str, Any]:
    strict = parse_generated_answer(str(row["raw_output"]))
    if strict.ok:
        prediction = strict
        parse_mode = "strict_final_marker"
    else:
        prediction = parse_last_numeric_answer(str(row["raw_output"]))
        parse_mode = "last_numeric_fallback" if prediction.ok else "failed"
    gold = Decimal(str(row["gold_value"]))
    expected = {
        "strict_parse_status": strict.status,
        "parse_mode": parse_mode,
        "parse_status": prediction.status,
        "parsed_prediction": (
            str(prediction.value) if prediction.value is not None else None
        ),
        "numeric_correct": bool(prediction.ok and prediction.value == gold),
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"stored OOD parser field changed for {row['record_id']}: {field}")
    return dict(row) | expected


def audit_complete_ood_dataset_from_raw(
    *, rows: list[dict[str, Any]], frozen_records: list[dict[str, Any]]
) -> dict[str, Any]:
    recomputed = [recompute_ood_generation_row(row) for row in rows]
    report = audit_complete_dataset(
        rows=recomputed,
        frozen_records=frozen_records,
    )
    report["parser_rows_recomputed_from_raw_text"] = len(recomputed)
    return report
