"""Optimizer-step planning for a fixed response-token exposure protocol."""

from __future__ import annotations

from collections.abc import Sequence

from eg_sft.training.effective_batch import TrainingItem


def balanced_optimizer_step_plan(
    *,
    epoch_orders: Sequence[Sequence[int]],
    optimizer_steps: int,
) -> list[tuple[TrainingItem, ...]]:
    """Partition all example occurrences into exactly ``optimizer_steps`` groups.

    The groups preserve the frozen epoch order and differ in sequence count by
    at most one. Losses inside a group must be summed over response tokens and
    normalized once by that group's response-token count.
    """

    if optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be positive")
    flattened = [
        TrainingItem(epoch=epoch, position=position, example_index=int(example_index))
        for epoch, order in enumerate(epoch_orders)
        for position, example_index in enumerate(order)
    ]
    if len(flattened) < optimizer_steps:
        raise ValueError("optimizer_steps cannot exceed example occurrences")
    base, remainder = divmod(len(flattened), optimizer_steps)
    groups: list[tuple[TrainingItem, ...]] = []
    cursor = 0
    for step_index in range(optimizer_steps):
        size = base + (1 if step_index < remainder else 0)
        groups.append(tuple(flattened[cursor : cursor + size]))
        cursor += size
    if cursor != len(flattened):  # pragma: no cover - defensive invariant
        raise AssertionError("optimizer step plan lost example occurrences")
    return groups


def micro_batches_for_step(
    step_items: Sequence[TrainingItem], *, micro_batch_size: int
) -> list[tuple[TrainingItem, ...]]:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if not step_items:
        raise ValueError("optimizer step cannot be empty")
    return [
        tuple(step_items[start : start + micro_batch_size])
        for start in range(0, len(step_items), micro_batch_size)
    ]


def optimizer_step_token_audit(
    *,
    step_token_counts: Sequence[int],
    expected_optimizer_steps: int,
    expected_exposure_tokens: int,
    tolerance_fraction: float,
) -> dict[str, float | int | bool]:
    if len(step_token_counts) != expected_optimizer_steps:
        raise ValueError("optimizer step token count length differs from protocol")
    if any(value <= 0 for value in step_token_counts):
        raise ValueError("each optimizer step must contain supervised response tokens")
    observed = sum(step_token_counts)
    relative_error = abs(observed - expected_exposure_tokens) / expected_exposure_tokens
    return {
        "optimizer_steps": len(step_token_counts),
        "response_supervision_exposure_tokens": observed,
        "target_response_supervision_exposure_tokens": expected_exposure_tokens,
        "exposure_relative_error": relative_error,
        "exposure_gate_passed": relative_error <= tolerance_fraction,
        "minimum_step_response_tokens": min(step_token_counts),
        "maximum_step_response_tokens": max(step_token_counts),
    }
