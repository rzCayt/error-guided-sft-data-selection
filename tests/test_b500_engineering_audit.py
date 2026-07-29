import pytest
import torch

from eg_sft.evaluation.resumable import aggregate_gsm8k_metrics
from eg_sft.experiment.b500_engineering_audit import (
    audit_completed_evaluation,
    summarize_adapter_tensors,
)


def _row(index: int, *, prompt_version: str = "frozen-v1") -> dict[str, object]:
    return {
        "record_id": f"record-{index}",
        "source_index": index,
        "question_sha256": f"question-{index}",
        "prompt_version": prompt_version,
        "numeric_correct": index == 0,
        "parse_status": "ok",
        "strict_parse_status": "ok",
        "parse_mode": "strict_final_marker",
    }


def _record(index: int) -> dict[str, object]:
    return {
        "record_id": f"record-{index}",
        "source_index": index,
        "question_sha256": f"question-{index}",
    }


def test_completed_evaluation_requires_exact_rows_and_metrics() -> None:
    rows = [_row(0), _row(1)]
    report = audit_completed_evaluation(
        rows=rows,
        frozen_records=[_record(0), _record(1)],
        metrics=aggregate_gsm8k_metrics(rows),
        prompt_version="frozen-v1",
    )
    assert report["row_count"] == 2
    assert report["unique_record_id_count"] == 2
    assert report["metrics_recomputed_exactly"] is True


def test_completed_evaluation_rejects_prompt_drift() -> None:
    with pytest.raises(ValueError, match="prompt version"):
        audit_completed_evaluation(
            rows=[_row(0, prompt_version="changed")],
            frozen_records=[_record(0)],
            metrics=aggregate_gsm8k_metrics([_row(0, prompt_version="changed")]),
            prompt_version="frozen-v1",
        )


def test_completed_evaluation_rejects_metric_drift() -> None:
    rows = [_row(0)]
    metrics = aggregate_gsm8k_metrics(rows)
    metrics["numeric_correct_count"] = 0
    with pytest.raises(ValueError, match="stored metric mismatch"):
        audit_completed_evaluation(
            rows=rows,
            frozen_records=[_record(0)],
            metrics=metrics,
            prompt_version="frozen-v1",
        )


def test_adapter_tensor_summary_requires_nonzero_lora_state() -> None:
    report = summarize_adapter_tensors(
        {
            "layer.lora_A.weight": torch.ones(2, 3),
            "layer.lora_B.weight": torch.tensor([[0.0, 1.0]]),
        }
    )
    assert report["tensor_count"] == 2
    assert report["total_parameters"] == 8
    assert report["nonzero_parameter_count"] == 7

    with pytest.raises(ValueError, match="all serialized"):
        summarize_adapter_tensors(
            {"layer.lora_A.weight": torch.zeros(2, 3)}
        )


def test_adapter_tensor_summary_rejects_non_lora_tensor() -> None:
    with pytest.raises(ValueError, match="non-LoRA"):
        summarize_adapter_tensors({"base.weight": torch.ones(2, 3)})
