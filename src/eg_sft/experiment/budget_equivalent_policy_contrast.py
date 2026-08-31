"""Accuracy-blind diagnostics for whether Phase 1 selection policies differ."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from eg_sft.selection.budget_equivalent import jaccard


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def source_diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["source_dataset"]) for row in rows)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("selected rows are empty")
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return {
        "source_counts": dict(sorted(counts.items())),
        "source_entropy_nats": entropy,
        "effective_source_count": math.exp(entropy),
        "dominant_source_fraction": max(probabilities),
    }


def selected_policy_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    rds_priority_by_id: Mapping[str, float],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("selected rows are empty")
    scores = [rds_priority_by_id[str(row["candidate_id"])] for row in rows]
    responses = [int(row["supervised_tokens"]) for row in rows]
    prompts = [int(row["total_tokens"]) - response for row, response in zip(rows, responses)]
    totals = [int(row["total_tokens"]) for row in rows]
    payload = {
        "selected_count": len(rows),
        "rds_rank_percentile": {
            "mean": statistics.fmean(scores),
            "median": statistics.median(scores),
            "q10": _quantile(scores, 0.10),
            "q90": _quantile(scores, 0.90),
            "minimum": min(scores),
            "maximum": max(scores),
        },
        "response_tokens": {
            "sum": sum(responses),
            "mean": statistics.fmean(responses),
            "median": statistics.median(responses),
        },
        "prompt_tokens": {
            "sum": sum(prompts),
            "mean": statistics.fmean(prompts),
            "median": statistics.median(prompts),
        },
        "total_nonpadding_tokens": {
            "sum": sum(totals),
            "mean": statistics.fmean(totals),
            "median": statistics.median(totals),
        },
    }
    return payload | source_diversity(rows)


def pair_policy_contrast(
    *,
    rds_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
    rds_priority_by_id: Mapping[str, float],
) -> dict[str, Any]:
    if len(rds_rows) != len(random_rows):
        raise ValueError("policy contrast lists must have equal size")
    rds_summary = selected_policy_summary(
        rows=rds_rows, rds_priority_by_id=rds_priority_by_id
    )
    random_summary = selected_policy_summary(
        rows=random_rows, rds_priority_by_id=rds_priority_by_id
    )
    rds_ids = [str(row["candidate_id"]) for row in rds_rows]
    random_ids = [str(row["candidate_id"]) for row in random_rows]
    overlap = jaccard(rds_ids, random_ids)
    rds_score = float(rds_summary["rds_rank_percentile"]["mean"])
    random_score = float(random_summary["rds_rank_percentile"]["mean"])
    return {
        "rds": rds_summary,
        "random": random_summary,
        "selected_id_jaccard": overlap,
        "replacement_fraction_of_budget": 1.0
        - len(set(rds_ids) & set(random_ids)) / len(rds_ids),
        "mean_rds_rank_percentile_lift": rds_score - random_score,
        "prompt_token_sum_difference": int(
            rds_summary["prompt_tokens"]["sum"]
            - random_summary["prompt_tokens"]["sum"]
        ),
        "total_token_sum_difference": int(
            rds_summary["total_nonpadding_tokens"]["sum"]
            - random_summary["total_nonpadding_tokens"]["sum"]
        ),
        "contrast_is_nonidentical": overlap < 1.0,
        "rds_score_direction_is_positive": rds_score > random_score,
    }


def common_stratum_contrast(
    *,
    rds_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
    rds_priority_by_id: Mapping[str, float],
    stratum_candidate_counts: Mapping[str, int],
) -> dict[str, Any]:
    rds_by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    random_by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    for row in rds_rows:
        rds_by_stratum.setdefault(str(row["common_mix_stratum"]), []).append(row)
    for row in random_rows:
        random_by_stratum.setdefault(str(row["common_mix_stratum"]), []).append(row)
    if {key: len(value) for key, value in rds_by_stratum.items()} != {
        key: len(value) for key, value in random_by_stratum.items()
    }:
        raise ValueError("common-mix quotas differ between policies")
    strata = []
    for key in sorted(rds_by_stratum):
        rds_scores = [rds_priority_by_id[str(row["candidate_id"])] for row in rds_by_stratum[key]]
        random_scores = [rds_priority_by_id[str(row["candidate_id"])] for row in random_by_stratum[key]]
        quota = len(rds_scores)
        available = int(stratum_candidate_counts[key])
        strata.append(
            {
                "stratum": key,
                "quota": quota,
                "available_candidates": available,
                "freedom_ratio": available / quota,
                "rds_mean_rank_percentile": statistics.fmean(rds_scores),
                "random_mean_rank_percentile": statistics.fmean(random_scores),
                "mean_rank_percentile_lift": statistics.fmean(rds_scores)
                - statistics.fmean(random_scores),
            }
        )
    return {
        "stratum_count": len(strata),
        "minimum_freedom_ratio": min(row["freedom_ratio"] for row in strata),
        "quota_weighted_mean_rank_percentile_lift": sum(
            row["quota"] * row["mean_rank_percentile_lift"] for row in strata
        )
        / sum(row["quota"] for row in strata),
        "strata": strata,
    }
