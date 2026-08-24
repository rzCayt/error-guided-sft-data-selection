import pytest

from eg_sft.experiment.budget_equivalent_audit import audit_training_artifacts


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
