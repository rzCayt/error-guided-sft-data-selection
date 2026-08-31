"""Budget-equivalent selection primitives for the v3 research protocol.

This module consumes frozen query-to-candidate similarities and token-audited
candidate metadata. It never loads a model or observes evaluation outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from eg_sft.selection.rds import rank_scores, round_robin_order
from eg_sft.training.b500 import selected_id_sha256


CORE_METHODS = (
    "random_free_mix",
    "rds_error_free_mix",
    "random_common_mix",
    "rds_error_common_mix",
)


def canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_priority(candidate_id: str, seed: int) -> float:
    """Map an ID and seed to a deterministic priority in (0, 1)."""

    digest = hashlib.sha256(f"{seed}\0{candidate_id}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / float(2**64)


def stratified_query_bootstrap_indices(
    query_inventory: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[int], list[int]]:
    """Sample correct and error queries separately while preserving group sizes."""

    correct = [i for i, row in enumerate(query_inventory) if not bool(row["is_error_query"])]
    errors = [i for i, row in enumerate(query_inventory) if bool(row["is_error_query"])]
    if not correct or not errors:
        raise ValueError("query inventory must contain correct and error queries")
    rng = random.Random(seed)
    sampled_correct = [rng.choice(correct) for _ in correct]
    sampled_errors = [rng.choice(errors) for _ in errors]
    all_indices = sampled_correct + sampled_errors
    rng.shuffle(all_indices)
    rng.shuffle(sampled_errors)
    return all_indices, sampled_errors


def bootstrap_rds_priorities(
    similarity: torch.Tensor,
    query_inventory: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Build complete all-query and error-query RDS rank priorities."""

    if similarity.ndim != 2:
        raise ValueError("similarity must be query-by-candidate")
    if similarity.shape[0] != len(query_inventory):
        raise ValueError("similarity query dimension differs from inventory")
    all_indices, error_indices = stratified_query_bootstrap_indices(
        query_inventory, seed=seed
    )
    all_order = round_robin_order(similarity[all_indices])
    error_order = round_robin_order(similarity[error_indices])
    candidate_count = similarity.shape[1]
    return (
        rank_scores(all_order, candidate_count=candidate_count),
        rank_scores(error_order, candidate_count=candidate_count),
        {
            "selection_replicate_seed": seed,
            "all_bootstrap_count": len(all_indices),
            "error_bootstrap_count": len(error_indices),
            "all_bootstrap_index_sha256": canonical_json_sha256(all_indices),
            "error_bootstrap_index_sha256": canonical_json_sha256(error_indices),
        },
    )


def response_length_thresholds(
    candidates: Sequence[Mapping[str, Any]], *, bin_count: int
) -> tuple[int, ...]:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    values = sorted(int(row["supervised_tokens"]) for row in candidates)
    if not values:
        raise ValueError("candidate inventory is empty")
    thresholds = []
    for boundary in range(1, bin_count):
        index = min(len(values) - 1, math.ceil(boundary * len(values) / bin_count) - 1)
        thresholds.append(values[index])
    return tuple(thresholds)


def response_length_bin(value: int, thresholds: Sequence[int]) -> int:
    return sum(value > threshold for threshold in thresholds)


def _largest_remainder_quotas(counts: Mapping[str, int], *, total: int) -> dict[str, int]:
    if total <= 0 or not counts or any(value <= 0 for value in counts.values()):
        raise ValueError("quota counts and total must be positive")
    population = sum(counts.values())
    exact = {key: total * value / population for key, value in counts.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


@dataclass(frozen=True)
class CommonMixDesign:
    source_groups: dict[str, str]
    response_thresholds: tuple[int, ...]
    stratum_quotas: dict[str, int]
    stratum_candidate_counts: dict[str, int]
    target_prompt_tokens: int
    target_total_tokens: int
    forced_selected_count: int
    selected_bin_count: int


def build_common_mix_design(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selection_count: int,
    target_response_tokens: int,
    requested_bin_count: int = 5,
    minimum_source_quota: int = 4,
    minimum_freedom_ratio: float = 4.0,
) -> CommonMixDesign:
    """Freeze source groups, length bins, quotas, and token targets."""

    if selection_count <= 0 or selection_count >= len(candidates):
        raise ValueError("selection_count must be positive and below pool size")
    source_counts = Counter(str(row["source_dataset"]) for row in candidates)
    source_groups = {
        source: (
            source
            if selection_count * count / len(candidates) >= minimum_source_quota
            else "other"
        )
        for source, count in source_counts.items()
    }
    for bin_count in range(requested_bin_count, 0, -1):
        thresholds = response_length_thresholds(candidates, bin_count=bin_count)
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in candidates:
            source = source_groups[str(row["source_dataset"])]
            length_bin = response_length_bin(int(row["supervised_tokens"]), thresholds)
            grouped[f"{source}|q{length_bin}"].append(row)
        counts = {key: len(rows) for key, rows in grouped.items()}
        quotas = _largest_remainder_quotas(counts, total=selection_count)
        if all(
            quota == 0 or counts[key] / quota >= minimum_freedom_ratio
            for key, quota in quotas.items()
        ):
            break
    else:  # pragma: no cover
        raise ValueError("could not build a feasible common-mix design")

    prompt_target = 0
    forced = 0
    for key, quota in quotas.items():
        if quota == 0:
            continue
        prompt_values = sorted(
            int(row["total_tokens"]) - int(row["supervised_tokens"])
            for row in grouped[key]
        )
        prompt_target += quota * int(round(statistics.median(prompt_values)))
        if len(grouped[key]) == quota:
            forced += quota
    return CommonMixDesign(
        source_groups=source_groups,
        response_thresholds=thresholds,
        stratum_quotas=quotas,
        stratum_candidate_counts=counts,
        target_prompt_tokens=prompt_target,
        target_total_tokens=prompt_target + target_response_tokens,
        forced_selected_count=forced,
        selected_bin_count=bin_count,
    )


def candidate_stratum(row: Mapping[str, Any], design: CommonMixDesign) -> str:
    source = design.source_groups[str(row["source_dataset"])]
    length_bin = response_length_bin(
        int(row["supervised_tokens"]), design.response_thresholds
    )
    return f"{source}|q{length_bin}"


def _cluster_values(
    candidates: Sequence[Mapping[str, Any]],
    *,
    duplicate_clusters: Mapping[str, str] | None,
    allow_exact_prompt_fallback: bool,
) -> tuple[list[str], str]:
    if duplicate_clusters is not None:
        missing = [
            str(row["candidate_id"])
            for row in candidates
            if str(row["candidate_id"]) not in duplicate_clusters
        ]
        if missing:
            raise ValueError(f"duplicate-cluster manifest misses {len(missing)} candidates")
        return (
            [duplicate_clusters[str(row["candidate_id"])] for row in candidates],
            "near_duplicate_cluster_manifest",
        )
    if not allow_exact_prompt_fallback:
        raise ValueError("formal selection requires a near-duplicate cluster manifest")
    return ([str(row["user_prompt_sha256"]) for row in candidates], "exact_prompt_fallback")


def solve_budget_equivalent_selection(
    candidates: Sequence[Mapping[str, Any]],
    priorities: Sequence[float],
    *,
    selection_count: int,
    target_response_tokens: int,
    response_tolerance_fraction: float,
    common_design: CommonMixDesign | None,
    prompt_tolerance_fraction: float,
    total_tolerance_fraction: float,
    duplicate_clusters: Mapping[str, str] | None = None,
    allow_exact_prompt_fallback: bool = False,
    time_limit_seconds: float = 120.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve one deterministic binary selection problem with SciPy/HiGHS."""

    if len(candidates) != len(priorities):
        raise ValueError("candidate and priority counts differ")
    if len(set(str(row["candidate_id"]) for row in candidates)) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("budget-equivalent selection requires numpy and scipy") from error

    clusters, cluster_mode = _cluster_values(
        candidates,
        duplicate_clusters=duplicate_clusters,
        allow_exact_prompt_fallback=allow_exact_prompt_fallback,
    )
    matrix_rows: list[list[float]] = [[1.0] * len(candidates)]
    lower = [float(selection_count)]
    upper = [float(selection_count)]
    response = [float(row["supervised_tokens"]) for row in candidates]
    response_delta = target_response_tokens * response_tolerance_fraction
    matrix_rows.append(response)
    lower.append(target_response_tokens - response_delta)
    upper.append(target_response_tokens + response_delta)

    if common_design is not None:
        prompts = [
            float(int(row["total_tokens"]) - int(row["supervised_tokens"]))
            for row in candidates
        ]
        totals = [float(row["total_tokens"]) for row in candidates]
        prompt_delta = common_design.target_prompt_tokens * prompt_tolerance_fraction
        total_delta = common_design.target_total_tokens * total_tolerance_fraction
        matrix_rows.extend((prompts, totals))
        lower.extend(
            (
                common_design.target_prompt_tokens - prompt_delta,
                common_design.target_total_tokens - total_delta,
            )
        )
        upper.extend(
            (
                common_design.target_prompt_tokens + prompt_delta,
                common_design.target_total_tokens + total_delta,
            )
        )
        strata = [candidate_stratum(row, common_design) for row in candidates]
        for key in sorted(common_design.stratum_quotas):
            quota = common_design.stratum_quotas[key]
            matrix_rows.append([1.0 if value == key else 0.0 for value in strata])
            lower.append(float(quota))
            upper.append(float(quota))

    cluster_indices: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        cluster_indices[cluster].append(index)
    for indices in cluster_indices.values():
        if len(indices) <= 1:
            continue
        constraint = [0.0] * len(candidates)
        for index in indices:
            constraint[index] = 1.0
        matrix_rows.append(constraint)
        lower.append(0.0)
        upper.append(1.0)

    result = milp(
        c=-np.asarray(priorities, dtype=np.float64),
        integrality=np.ones(len(candidates), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            csr_matrix(np.asarray(matrix_rows, dtype=np.float64)),
            np.asarray(lower),
            np.asarray(upper),
        ),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.0},
    )
    if result.x is None or not bool(result.success):
        raise RuntimeError(
            f"selection MILP failed: status={result.status}, message={result.message}"
        )
    selected_indices = [index for index, value in enumerate(result.x) if value >= 0.5]
    if len(selected_indices) != selection_count:
        raise ValueError("MILP returned an unexpected selected count")
    selected_indices.sort(
        key=lambda index: (-priorities[index], str(candidates[index]["candidate_id"]))
    )
    selected = []
    for rank, index in enumerate(selected_indices):
        row = dict(candidates[index])
        row["selection_rank"] = rank
        row["selection_priority"] = float(priorities[index])
        if common_design is not None:
            row["common_mix_stratum"] = candidate_stratum(row, common_design)
        selected.append(row)

    response_sum = sum(int(row["supervised_tokens"]) for row in selected)
    prompt_sum = sum(
        int(row["total_tokens"]) - int(row["supervised_tokens"]) for row in selected
    )
    total_sum = sum(int(row["total_tokens"]) for row in selected)
    audit = {
        "solver": "scipy.optimize.milp_highs",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(-result.fun),
        "selected_count": len(selected),
        "response_supervision_tokens": response_sum,
        "response_target_tokens": target_response_tokens,
        "response_relative_error": abs(response_sum - target_response_tokens)
        / target_response_tokens,
        "prompt_tokens": prompt_sum,
        "total_nonpadding_tokens": total_sum,
        "duplicate_cluster_mode": cluster_mode,
        "selected_id_sha256": selected_id_sha256(selected),
    }
    if common_design is not None:
        observed = Counter(str(row["common_mix_stratum"]) for row in selected)
        audit.update(
            {
                "common_mix_quota_matches": dict(observed)
                == {key: quota for key, quota in common_design.stratum_quotas.items() if quota > 0},
                "common_mix_observed_quotas": dict(sorted(observed.items())),
                "target_prompt_tokens": common_design.target_prompt_tokens,
                "target_total_tokens": common_design.target_total_tokens,
                "prompt_relative_error": abs(prompt_sum - common_design.target_prompt_tokens)
                / common_design.target_prompt_tokens,
                "total_relative_error": abs(total_sum - common_design.target_total_tokens)
                / common_design.target_total_tokens,
            }
        )
    return selected, audit


def build_selection_manifest(
    *,
    method: str,
    selection_seed: int,
    train_seed: int,
    selected: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if method not in CORE_METHODS:
        raise ValueError(f"unsupported core method: {method}")
    manifest: dict[str, Any] = {
        "schema_version": "budget-equivalent-selection-v3",
        "strategy": method,
        "budget": len(selected),
        "selection_seed": selection_seed,
        "train_seed": train_seed,
        "selected_id_sha256": selected_id_sha256(selected),
        "selected_candidates": [dict(row) for row in selected],
        "budget_audit": dict(audit),
        "provenance": dict(provenance),
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def median_pairwise_jaccard(id_sets: Sequence[Sequence[str]]) -> float:
    values = [
        jaccard(id_sets[left], id_sets[right])
        for left in range(len(id_sets))
        for right in range(left + 1, len(id_sets))
    ]
    if not values:
        raise ValueError("at least two selection replicates are required")
    return float(statistics.median(values))
