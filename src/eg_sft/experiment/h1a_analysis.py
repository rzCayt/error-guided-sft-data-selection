"""Preregistered H1a statistics for error-conditioned RDS+ scores."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch


def average_ranks(values: Sequence[float]) -> torch.Tensor:
    """Return one-based average ranks, including deterministic tie handling."""

    count = len(values)
    if count == 0:
        raise ValueError("cannot rank an empty sequence")
    order = sorted(range(count), key=lambda index: (values[index], index))
    ranks = [0.0] * count
    start = 0
    while start < count:
        end = start + 1
        while end < count and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return torch.tensor(ranks, dtype=torch.float64)


def _residualize(
    outcome: torch.Tensor,
    controls: Sequence[torch.Tensor],
) -> torch.Tensor:
    columns = [torch.ones_like(outcome), *controls]
    design = torch.stack(columns, dim=1)
    coefficients = torch.linalg.lstsq(design, outcome).solution
    return outcome - design @ coefficients


def partial_spearman(
    *,
    predictor: Sequence[float],
    outcome: Sequence[float],
    controls: Sequence[Sequence[float]],
) -> float:
    """Spearman correlation after rank-residualizing both focal variables."""

    if len(predictor) != len(outcome) or len(predictor) < 4:
        raise ValueError("predictor and outcome lengths must match and be at least four")
    if any(len(control) != len(outcome) for control in controls):
        raise ValueError("control lengths must match the outcome")

    predictor_rank = average_ranks(predictor)
    outcome_rank = average_ranks(outcome)
    control_ranks = [average_ranks(control) for control in controls]
    predictor_residual = _residualize(predictor_rank, control_ranks)
    outcome_residual = _residualize(outcome_rank, control_ranks)
    denominator = torch.sqrt(
        torch.sum(predictor_residual**2) * torch.sum(outcome_residual**2)
    )
    if float(denominator) <= 1e-15:
        # With no rank variation left after the controls, the focal score
        # contains no measurable incremental ordering information.  Defining
        # the statistic as zero keeps that preregistered null case in the
        # permutation distribution instead of silently dropping it.
        return 0.0
    return float(
        torch.sum(predictor_residual * outcome_residual) / denominator
    )


def fixed_count_permutations(
    *,
    item_count: int,
    selected_count: int,
    permutation_count: int,
    seed: int,
) -> list[list[int]]:
    """Draw deterministic fixed-size pseudo-error index sets."""

    if not 0 < selected_count < item_count:
        raise ValueError("selected_count must be between zero and item_count")
    if permutation_count <= 0:
        raise ValueError("permutation_count must be positive")
    generator = random.Random(seed)
    population = list(range(item_count))
    return [
        sorted(generator.sample(population, selected_count))
        for _ in range(permutation_count)
    ]


def top_bottom_mean_difference(
    *,
    scores: Sequence[float],
    utilities: Sequence[float],
    group_count: int,
) -> tuple[float, list[int], list[int]]:
    """Return top-score mean utility minus bottom-score mean utility."""

    if len(scores) != len(utilities):
        raise ValueError("scores and utilities lengths differ")
    if group_count <= 0 or 2 * group_count > len(scores):
        raise ValueError("invalid top/bottom group count")
    order = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )
    top = order[:group_count]
    bottom = order[-group_count:]
    top_mean = sum(utilities[index] for index in top) / group_count
    bottom_mean = sum(utilities[index] for index in bottom) / group_count
    return top_mean - bottom_mean, top, bottom
