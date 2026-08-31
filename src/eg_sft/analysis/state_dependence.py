"""Statistics for candidate-utility reliability and cross-state transfer."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence

from eg_sft.analysis.behavior_composition import spearman
from eg_sft.experiment.utility import icc_absolute_agreement


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_interval(
    *,
    sample_size: int,
    statistic: Callable[[list[int]], float],
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    if sample_size < 2:
        raise ValueError("bootstrap requires at least two candidates")
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        indices = [rng.randrange(sample_size) for _ in range(sample_size)]
        estimate = float(statistic(indices))
        if not math.isfinite(estimate):
            raise ValueError("bootstrap statistic produced a non-finite value")
        estimates.append(estimate)
    return {
        "repetitions": repetitions,
        "seed": seed,
        "lower_95": percentile(estimates, 0.025),
        "median": percentile(estimates, 0.5),
        "upper_95": percentile(estimates, 0.975),
    }


def top_k_jaccard(
    candidate_ids: Sequence[str], left: Sequence[float], right: Sequence[float], *, k: int
) -> float:
    if len(candidate_ids) != len(left) or len(left) != len(right):
        raise ValueError("top-k inputs must have equal length")
    if k <= 0 or k >= len(candidate_ids):
        raise ValueError("k must be positive and smaller than candidate count")
    left_top = set(
        sorted(zip(candidate_ids, left, strict=True), key=lambda item: (-item[1], item[0]))[:k]
    )
    right_top = set(
        sorted(zip(candidate_ids, right, strict=True), key=lambda item: (-item[1], item[0]))[:k]
    )
    left_ids = {item[0] for item in left_top}
    right_ids = {item[0] for item in right_top}
    return len(left_ids & right_ids) / len(left_ids | right_ids)


def pairwise_spearman(by_seed: dict[int, Sequence[float]]) -> dict[str, float]:
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    return {
        f"{left}_{right}": spearman(by_seed[left], by_seed[right])
        for index, left in enumerate(seeds)
        for right in seeds[index + 1 :]
    }


def u0_point_metrics(matrix: Sequence[Sequence[float]], seeds: Sequence[int]) -> dict:
    if not matrix or any(len(row) != len(seeds) for row in matrix):
        raise ValueError("matrix shape differs from seed count")
    by_seed = {
        int(seed): [float(row[index]) for row in matrix]
        for index, seed in enumerate(seeds)
    }
    pairwise = pairwise_spearman(by_seed)
    within_candidate_sd = [statistics.stdev(row) for row in matrix]
    return {
        "icc_absolute_agreement_a1": icc_absolute_agreement(matrix),
        "pairwise_spearman": pairwise,
        "median_pairwise_spearman": statistics.median(pairwise.values()),
        "minimum_pairwise_spearman": min(pairwise.values()),
        "median_within_candidate_sd": statistics.median(within_candidate_sd),
    }


def correlation_with_interval(
    left: Sequence[float],
    right: Sequence[float],
    *,
    repetitions: int,
    seed: int,
) -> dict:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must have equal length >= 2")
    point = spearman(left, right)
    interval = bootstrap_interval(
        sample_size=len(left),
        statistic=lambda indices: spearman(
            [left[index] for index in indices], [right[index] for index in indices]
        ),
        repetitions=repetitions,
        seed=seed,
    )
    return {"point": point, "bootstrap_95": interval}
