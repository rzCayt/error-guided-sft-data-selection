"""Candidate-utility measurement and test-retest reliability statistics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader


def causal_supervised_token_count(labels: torch.Tensor) -> int:
    """Count labels consumed by Transformers' causal-LM shifted loss."""

    if labels.ndim < 2 or labels.shape[-1] < 2:
        raise ValueError("labels must have batch and sequence dimensions")
    return int((labels[..., 1:] != -100).sum().item())


def to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.no_grad()
def mean_supervised_token_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute token-weighted response-only loss."""

    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    for batch in loader:
        batch = to_device(batch, device)
        token_count = causal_supervised_token_count(batch["labels"])
        if token_count == 0:
            raise ValueError("evaluation batch has zero supervised tokens")
        loss = model(**batch).loss
        weighted_loss += float(loss.item()) * token_count
        supervised_tokens += token_count
    if supervised_tokens == 0:
        raise ValueError("evaluation has zero supervised tokens")
    return weighted_loss / supervised_tokens


def icc_absolute_agreement(
    measurements: Sequence[Sequence[float]],
) -> float:
    """Two-way mixed, absolute-agreement, single-measure ICC(A,1).

    Rows are candidates and columns are repeated measurements. This statistic
    penalizes both candidate-specific noise and a systematic shift between
    repeats.
    """

    values = torch.tensor(measurements, dtype=torch.float64)
    if values.ndim != 2:
        raise ValueError("measurements must be a two-dimensional matrix")
    target_count, repeat_count = values.shape
    if target_count < 2 or repeat_count < 2:
        raise ValueError("ICC requires at least two targets and two repeats")
    if not torch.isfinite(values).all():
        raise ValueError("measurements must be finite")

    grand_mean = values.mean()
    target_means = values.mean(dim=1)
    repeat_means = values.mean(dim=0)
    ms_targets = (
        repeat_count
        * torch.sum((target_means - grand_mean) ** 2)
        / (target_count - 1)
    )
    ms_repeats = (
        target_count
        * torch.sum((repeat_means - grand_mean) ** 2)
        / (repeat_count - 1)
    )
    residual = (
        values
        - target_means.unsqueeze(1)
        - repeat_means.unsqueeze(0)
        + grand_mean
    )
    ms_error = torch.sum(residual**2) / (
        (target_count - 1) * (repeat_count - 1)
    )
    denominator = (
        ms_targets
        + (repeat_count - 1) * ms_error
        + repeat_count * (ms_repeats - ms_error) / target_count
    )
    if float(denominator) == 0.0:
        raise ValueError("ICC denominator is zero")
    return float((ms_targets - ms_error) / denominator)


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal sequences of length at least two")
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(torch.sum(x**2) * torch.sum(y**2))
    if float(denominator) == 0.0:
        raise ValueError("correlation denominator is zero")
    return float(torch.sum(x * y) / denominator)
