from eg_sft.experiment.budget_equivalent_blind_aggregate import (
    guarded_blind_aggregation,
)


def _private_map() -> dict:
    return {
        "required_audited_cells_before_unblinding": 2,
        "cells": [
            {
                "blind_cell_id": "cell_001",
                "cell_id": "actual_a",
                "method_alias": "method_A",
                "replicate_index": 1,
                "train_seed": 17,
            },
            {
                "blind_cell_id": "cell_002",
                "cell_id": "actual_b",
                "method_alias": "method_B",
                "replicate_index": 1,
                "train_seed": 17,
            },
        ],
    }


def test_aggregate_gate_requires_formal_and_ood_audits() -> None:
    registry = {
        "jobs": [
            {"cell_id": "actual_a", "status": "AUDITED_PASS"},
            {"cell_id": "actual_b", "status": "AUDITED_PASS"},
        ]
    }
    blocked = guarded_blind_aggregation(
        private_map=_private_map(),
        registry=registry,
        ood_status_by_cell={"actual_a": "AUDITED_PASS", "actual_b": "PENDING"},
        ood_required=True,
    )
    assert blocked["status"] == "BLOCKED_INCOMPLETE"
    assert blocked["unblinding_permitted"] is False
    ready = guarded_blind_aggregation(
        private_map=_private_map(),
        registry=registry,
        ood_status_by_cell={
            "actual_a": "AUDITED_PASS",
            "actual_b": "AUDITED_PASS",
        },
        ood_required=True,
    )
    assert ready["status"] == "READY_FOR_SEPARATE_UNBLINDING"
    assert ready["unblinding_permitted"] is True
    assert ready["accuracy_withheld"] is True


def test_aggregate_output_hides_actual_cell_and_method_names() -> None:
    payload = guarded_blind_aggregation(
        private_map=_private_map(),
        registry={
            "jobs": [
                {"cell_id": "actual_a", "status": "PENDING"},
                {"cell_id": "actual_b", "status": "PENDING"},
            ]
        },
        ood_status_by_cell={},
        ood_required=True,
    )
    text = str(payload)
    assert "actual_a" not in text
    assert "actual_b" not in text
    assert "numeric_correct" not in text
