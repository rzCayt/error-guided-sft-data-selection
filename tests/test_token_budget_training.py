import pytest

from eg_sft.training.token_budget import (
    balanced_optimizer_step_plan,
    micro_batches_for_step,
    optimizer_step_token_audit,
)


def test_one_thousand_occurrences_are_partitioned_into_exactly_sixty_four_steps() -> None:
    plan = balanced_optimizer_step_plan(
        epoch_orders=[list(range(500)), list(reversed(range(500)))],
        optimizer_steps=64,
    )
    assert len(plan) == 64
    assert sorted(len(group) for group in plan) == [15] * 24 + [16] * 40
    flattened = [item.example_index for group in plan for item in group]
    assert flattened[:500] == list(range(500))
    assert flattened[500:] == list(reversed(range(500)))


def test_micro_batches_never_cross_optimizer_step_boundary() -> None:
    plan = balanced_optimizer_step_plan(
        epoch_orders=[list(range(10)), list(range(10))],
        optimizer_steps=4,
    )
    batches = [micro_batches_for_step(group, micro_batch_size=2) for group in plan]
    assert [[len(batch) for batch in step] for step in batches] == [
        [2, 2, 1],
        [2, 2, 1],
        [2, 2, 1],
        [2, 2, 1],
    ]


def test_step_plan_rejects_more_updates_than_occurrences() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        balanced_optimizer_step_plan(epoch_orders=[[0, 1]], optimizer_steps=3)


def test_exposure_audit_reports_gate_without_hiding_error() -> None:
    audit = optimizer_step_token_audit(
        step_token_counts=[1000] * 64,
        expected_optimizer_steps=64,
        expected_exposure_tokens=64_000,
        tolerance_fraction=0.005,
    )
    assert audit["exposure_gate_passed"] is True
    assert audit["exposure_relative_error"] == 0.0
