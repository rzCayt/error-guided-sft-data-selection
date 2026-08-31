import hashlib
from copy import deepcopy

import pytest

from eg_sft.experiment.budget_equivalent_audit import (
    audit_dose_only_token_cap_artifacts,
    audit_training_artifacts,
)


def _fixture() -> tuple[dict, dict, list[dict], list[dict]]:
    step_rows = [
        {"optimizer_step": index, "response_supervision_tokens": 1000}
        for index in range(1, 65)
    ]
    metrics = {
        "status": "PASS",
        "selected_count": 500,
        "optimizer_steps_planned": 64,
        "optimizer_steps_completed": 64,
        "adapter_reload_loss_absolute_difference": 0.0,
        "supervised_tokens_seen": 64000,
    }
    budget = {
        "exposure_gate_passed": True,
        "optimizer_steps": 64,
        "response_supervision_exposure_tokens": 64000,
    }
    token_audit = [{"candidate_id": f"c{index}"} for index in range(500)]
    return metrics, budget, token_audit, step_rows


def test_training_audit_requires_all_sixty_four_token_normalized_steps() -> None:
    metrics, budget, token_audit, step_rows = _fixture()
    report = audit_training_artifacts(
        training_metrics=metrics,
        token_budget_audit=budget,
        token_audit=token_audit,
        optimizer_step_rows=step_rows,
    )
    assert report["optimizer_steps_completed"] == 64
    assert report["token_budget_gate_passed"] is True


def test_training_audit_rejects_missing_step_even_when_metrics_claim_pass() -> None:
    metrics, budget, token_audit, step_rows = _fixture()
    with pytest.raises(ValueError, match="64 rows"):
        audit_training_artifacts(
            training_metrics=metrics,
            token_budget_audit=budget,
            token_audit=token_audit,
            optimizer_step_rows=step_rows[:-1],
        )


def _dose_fixture() -> tuple[dict, dict, list[dict]]:
    selected_sha = hashlib.sha256(b"selected-token-set").hexdigest()
    mask_shas = [
        hashlib.sha256(f"mask-{index}".encode()).hexdigest()
        for index in range(1, 65)
    ]
    rows = [
        {
            "optimizer_step": index,
            "response_supervision_tokens": 995,
            "candidate_response_supervision_tokens": 1005,
            "kept_response_supervision_tokens": 995,
            "token_cap_mask_sha256": mask_shas[index - 1],
            "cumulative_response_supervision_tokens": index * 995,
            "token_cap_policy": "hash_uniform_v1",
            "selected_token_set_sha256": selected_sha,
            "legacy_sequence_step_boundaries_preserved": False,
            "boundary_split_occurrence_count": 61,
            "selected_candidate_id_coverage": 500,
            "candidate_id_count": 500,
            "occurrence_with_kept_token_count": 1000,
            "occurrence_count": 1000,
            "mandatory_coverage_token_count": 1000,
        }
        for index in range(1, 65)
    ]
    shared = {
        "supervision_token_cap": 63680,
        "supervision_tokens_per_optimizer_step": 995,
        "token_cap_policy": "hash_uniform_v1",
        "selected_token_set_sha256": selected_sha,
        "legacy_sequence_step_boundaries_preserved": False,
        "boundary_split_occurrence_count": 61,
        "selected_candidate_id_coverage": 500,
        "candidate_id_count": 500,
        "occurrence_with_kept_token_count": 1000,
        "occurrence_count": 1000,
        "mandatory_coverage_token_count": 1000,
    }
    metrics = shared | {"supervised_tokens_seen": 63680}
    mask_set_text = "\n".join(mask_shas) + "\n"
    budget = shared | {
        "response_supervision_exposure_tokens": 63680,
        "optimizer_step_mask_set_sha256": hashlib.sha256(
            mask_set_text.encode()
        ).hexdigest(),
    }
    return metrics, budget, rows


def test_dose_only_audit_accepts_complete_mask_and_coverage_evidence() -> None:
    metrics, budget, rows = _dose_fixture()
    report = audit_dose_only_token_cap_artifacts(
        training_metrics=metrics,
        token_budget_audit=budget,
        optimizer_step_rows=rows,
        supervision_token_cap=63680,
        token_cap_policy="hash_uniform_v1",
    )
    assert report["status"] == "PASS"
    assert report["occurrence_with_kept_token_count"] == 1000


@pytest.mark.parametrize("corruption", ["mask_sha", "token_count"])
def test_dose_only_audit_rejects_step_corruption(corruption: str) -> None:
    metrics, budget, rows = _dose_fixture()
    corrupted = deepcopy(rows)
    if corruption == "mask_sha":
        corrupted[0]["token_cap_mask_sha256"] = "0" * 64
    else:
        corrupted[0]["response_supervision_tokens"] = 999
    with pytest.raises(ValueError, match="dose-only"):
        audit_dose_only_token_cap_artifacts(
            training_metrics=metrics,
            token_budget_audit=budget,
            optimizer_step_rows=corrupted,
            supervision_token_cap=63680,
            token_cap_policy="hash_uniform_v1",
        )
