"""Deterministic checks for a completed B=500 engineering evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from eg_sft.evaluation.resumable import (
    aggregate_gsm8k_metrics,
    validate_completed_prefix,
)


def audit_completed_evaluation(
    *,
    rows: Sequence[dict[str, Any]],
    frozen_records: Sequence[dict[str, Any]],
    metrics: Mapping[str, Any],
    prompt_version: str,
) -> dict[str, Any]:
    """Require a complete, ordered, uniquely keyed and reproducible evaluation."""

    if len(rows) != len(frozen_records):
        raise ValueError(
            f"evaluation row count changed: {len(rows)} != {len(frozen_records)}"
        )
    validate_completed_prefix(
        completed_rows=rows,
        frozen_records=frozen_records,
    )
    record_ids = [str(row["record_id"]) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("evaluation contains duplicate record IDs")

    for index, (row, frozen) in enumerate(
        zip(rows, frozen_records, strict=True)
    ):
        if int(row["source_index"]) != int(frozen["source_index"]):
            raise ValueError(f"source index mismatch at evaluation row {index}")
        if row["question_sha256"] != frozen["question_sha256"]:
            raise ValueError(f"question hash mismatch at evaluation row {index}")
        if row["prompt_version"] != prompt_version:
            raise ValueError(f"prompt version mismatch at evaluation row {index}")

    recomputed = aggregate_gsm8k_metrics(rows)
    for key, value in recomputed.items():
        if metrics.get(key) != value:
            raise ValueError(
                f"stored metric mismatch for {key}: "
                f"{metrics.get(key)!r} != {value!r}"
            )

    return {
        "row_count": len(rows),
        "unique_record_id_count": len(set(record_ids)),
        "ordered_frozen_prefix": True,
        "source_indexes_match": True,
        "question_hashes_match": True,
        "prompt_version": prompt_version,
        "metrics_recomputed_exactly": True,
        "recomputed_metrics": recomputed,
    }


def summarize_adapter_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Check that a serialized adapter contains non-empty LoRA tensors."""

    if not tensors:
        raise ValueError("adapter state contains no tensors")
    unexpected = sorted(name for name in tensors if "lora_" not in name)
    if unexpected:
        raise ValueError(
            "adapter state contains non-LoRA tensors: " + ", ".join(unexpected)
        )

    total_parameters = sum(int(tensor.numel()) for tensor in tensors.values())
    nonzero_tensor_count = sum(
        bool(torch.count_nonzero(tensor).item()) for tensor in tensors.values()
    )
    nonzero_parameter_count = sum(
        int(torch.count_nonzero(tensor).item()) for tensor in tensors.values()
    )
    if total_parameters <= 0:
        raise ValueError("adapter state contains no parameters")
    if nonzero_parameter_count <= 0:
        raise ValueError("all serialized adapter parameters are zero")

    return {
        "tensor_count": len(tensors),
        "total_parameters": total_parameters,
        "nonzero_tensor_count": nonzero_tensor_count,
        "nonzero_parameter_count": nonzero_parameter_count,
        "all_tensor_names_are_lora": True,
    }
