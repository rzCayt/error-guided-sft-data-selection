"""Fail-closed aggregation gate for blinded Phase 1 results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eg_sft.experiment.budget_equivalent_blind import blinded_registry


def guarded_blind_aggregation(
    *,
    private_map: Mapping[str, Any],
    registry: Mapping[str, Any],
    ood_status_by_cell: Mapping[str, str],
    ood_required: bool,
) -> dict[str, Any]:
    """Expose progress only; never emit accuracy or actual method names."""

    blind = blinded_registry(private_map=private_map, registry=registry)
    actual_by_blind = {
        str(row["blind_cell_id"]): str(row["cell_id"])
        for row in private_map["cells"]
    }
    rows = []
    for row in blind["jobs"]:
        actual = actual_by_blind[str(row["blind_cell_id"])]
        ood_status = str(ood_status_by_cell.get(actual, "PENDING"))
        rows.append(dict(row) | {"ood_audit_status": ood_status})
    formal_count = sum(row["status"] == "AUDITED_PASS" for row in rows)
    ood_count = sum(row["ood_audit_status"] == "AUDITED_PASS" for row in rows)
    required = len(rows)
    ready = formal_count == required and (not ood_required or ood_count == required)
    return {
        "schema_version": "budget-equivalent-phase1-blind-aggregate-gate-v1",
        "status": "READY_FOR_SEPARATE_UNBLINDING" if ready else "BLOCKED_INCOMPLETE",
        "job_count": required,
        "formal_audited_pass_count": formal_count,
        "ood_audited_pass_count": ood_count,
        "ood_required": ood_required,
        "unblinding_permitted": ready,
        "accuracy_withheld": True,
        "actual_method_names_withheld": True,
        "jobs": rows,
    }
