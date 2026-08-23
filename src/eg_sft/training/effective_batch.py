"""Deterministic micro-batching with response-token-normalized gradients."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrainingItem:
    """One example occurrence in a frozen multi-epoch training order."""

    epoch: int
    position: int
    example_index: int


def validate_micro_batch_contract(
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    nominal_effective_batch_size: int,
) -> None:
    """Require a positive micro-batch pair with the frozen effective batch."""

    values = (
        micro_batch_size,
        gradient_accumulation_steps,
        nominal_effective_batch_size,
    )
    if any(value <= 0 for value in values):
        raise ValueError("batch sizes and accumulation steps must be positive")
    observed = micro_batch_size * gradient_accumulation_steps
    if observed != nominal_effective_batch_size:
        raise ValueError(
            "micro_batch_size * gradient_accumulation_steps must equal "
            f"nominal_effective_batch_size ({observed} != "
            f"{nominal_effective_batch_size})"
        )


def build_training_micro_batches(
    *,
    epoch_orders: Sequence[Sequence[int]],
    micro_batch_size: int,
) -> list[tuple[TrainingItem, ...]]:
    """Flatten frozen epochs, then form batches without losing boundary items."""

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    flattened = [
        TrainingItem(epoch=epoch, position=position, example_index=int(example_index))
        for epoch, order in enumerate(epoch_orders)
        for position, example_index in enumerate(order)
    ]
    if not flattened:
        raise ValueError("epoch_orders must contain at least one example")
    return [
        tuple(flattened[start : start + micro_batch_size])
        for start in range(0, len(flattened), micro_batch_size)
    ]


def optimizer_steps_for_examples(
    *,
    example_count: int,
    nominal_effective_batch_size: int,
) -> int:
    if example_count <= 0 or nominal_effective_batch_size <= 0:
        raise ValueError("example_count and effective batch must be positive")
    return math.ceil(example_count / nominal_effective_batch_size)


def shifted_response_loss_sums(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable per-example loss sums and supervised token counts.

    Causal language-model labels are shifted once here. The caller accumulates
    the returned loss sums across all micro-batches in one effective batch and
    divides gradients once by the total returned token count.
    """

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence dimensions differ")
    if logits.shape[1] < 2:
        raise ValueError("causal loss requires at least two sequence positions")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    token_counts = valid.sum(dim=1)
    if int(token_counts.sum().item()) == 0:
        raise ValueError("batch has zero shifted supervised response tokens")
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        reduction="none",
        ignore_index=ignore_index,
    )
    loss_sums = (token_losses * valid.to(token_losses.dtype)).sum(dim=1)
    return loss_sums, token_counts


def normalize_gradients_by_token_count(
    parameters: Iterable[torch.nn.Parameter | torch.Tensor],
    *,
    response_token_count: int,
) -> None:
    """Normalize accumulated loss-sum gradients once per optimizer update."""

    if response_token_count <= 0:
        raise ValueError("response_token_count must be positive")
    scale = 1.0 / float(response_token_count)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def should_write_checkpoint(
    *,
    optimizer_step: int,
    optimizer_steps_planned: int,
    checkpoint_every_optimizer_steps: int,
) -> bool:
    """Save at the configured interval and always at the final update."""

    if optimizer_step <= 0 or optimizer_steps_planned <= 0:
        raise ValueError("optimizer steps must be positive")
    if checkpoint_every_optimizer_steps <= 0:
        raise ValueError("checkpoint interval must be positive")
    if optimizer_step > optimizer_steps_planned:
        raise ValueError("optimizer_step exceeds the frozen plan")
    return (
        optimizer_step % checkpoint_every_optimizer_steps == 0
        or optimizer_step == optimizer_steps_planned
    )
