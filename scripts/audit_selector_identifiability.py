from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.selection.error_guided import (  # noqa: E402
    _stable_noise,
    build_profile_weights,
    select_error_guided,
)
from eg_sft.selection.matched_random import matching_key, select_matched_random  # noqa: E402
from eg_sft.utils.io import read_csv, read_jsonl, write_csv  # noqa: E402

AUDIT_VERSION = "selector_identifiability_audit_v2"
MATCHING_FIELDS = (
    "task_family",
    "difficulty_bucket",
    "answer_magnitude_bucket",
    "reasoning_length_bucket",
)
PROFILE_SIGNAL_FIELDS = MATCHING_FIELDS
INSTANCE_LEVEL_SIGNAL_FIELDS: tuple[str, ...] = ()
TIE_BREAKER_FIELDS = ("id", "seed")
NULL_REPLICATES = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(rows: list[dict]) -> Counter:
    return Counter(matching_key(row) for row in rows)


def _distribution_l1(left: list[dict], right: list[dict]) -> int:
    left_dist = _distribution(left)
    right_dist = _distribution(right)
    return sum(
        abs(left_dist.get(key, 0) - right_dist.get(key, 0))
        for key in set(left_dist) | set(right_dist)
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _select_with_fixed_quotas(
    candidates: list[dict],
    quotas: Counter,
    seed: int,
    method: str,
) -> list[dict]:
    by_stratum: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        by_stratum[matching_key(row)].append(row)

    rng = random.Random(seed)
    selected: list[dict] = []
    for key, count in sorted(quotas.items()):
        pool = list(by_stratum[key])
        if method == "hash":
            pool.sort(key=lambda row: _stable_noise(row["id"], seed), reverse=True)
        elif method == "random":
            rng.shuffle(pool)
        else:
            raise ValueError(f"unknown fixed-quota method: {method}")
        selected.extend(pool[:count])
    return selected


def _pairwise_jaccard_rows(
    selections: dict[int, list[dict]],
    label: str,
) -> tuple[list[dict[str, object]], list[float]]:
    rows: list[dict[str, object]] = []
    values: list[float] = []
    for left_seed, right_seed in itertools.combinations(sorted(selections), 2):
        left_ids = {row["id"] for row in selections[left_seed]}
        right_ids = {row["id"] for row in selections[right_seed]}
        value = _jaccard(left_ids, right_ids)
        values.append(value)
        rows.append(
            {
                "comparison": label,
                "left_seed": left_seed,
                "right_seed": right_seed,
                "selected_id_jaccard": round(value, 6),
            }
        )
    return rows, values


def analyze_selector_identifiability(
    candidates: list[dict],
    profile_rows: list[dict[str, str]],
    budget: int,
    seeds: list[int],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(candidates):
        raise ValueError("budget cannot exceed candidate count")
    if not seeds:
        raise ValueError("at least one seed is required")

    weights = build_profile_weights(profile_rows)
    by_stratum: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        by_stratum[matching_key(row)].append(row)

    stratum_rows: list[dict[str, object]] = []
    max_weight_variants = 0
    covered_candidates = 0
    for key, rows in sorted(by_stratum.items()):
        candidate_weights = {weights.get(key, 1.0) for _ in rows}
        max_weight_variants = max(max_weight_variants, len(candidate_weights))
        if key in weights:
            covered_candidates += len(rows)
        stratum_rows.append(
            {
                "task_family": key[0],
                "difficulty_bucket": key[1],
                "answer_magnitude_bucket": key[2],
                "reasoning_length_bucket": key[3],
                "candidate_count": len(rows),
                "profile_covered": key in weights,
                "profile_weight": round(weights.get(key, 1.0), 6),
                "unique_nonrandom_weight_count_within_stratum": len(candidate_weights),
                "has_instance_level_signal_within_stratum": False,
                "within_stratum_order_source": "id_seed_hash_only",
            }
        )

    selections = {
        seed: select_error_guided(candidates, profile_rows, budget=budget, seed=seed)
        for seed in seeds
    }
    primary_seed = seeds[0]
    targeted = selections[primary_seed]
    matched = select_matched_random(candidates, targeted, seed=primary_seed + 1)
    targeted_ids = {row["id"] for row in targeted}
    matched_ids = {row["id"] for row in matched}

    seed_rows: list[dict[str, object]] = []
    pair_jaccards: list[float] = []
    for left_seed, right_seed in itertools.combinations(seeds, 2):
        left = selections[left_seed]
        right = selections[right_seed]
        jaccard = _jaccard({row["id"] for row in left}, {row["id"] for row in right})
        pair_jaccards.append(jaccard)
        seed_rows.append(
            {
                "left_seed": left_seed,
                "right_seed": right_seed,
                "selected_id_jaccard": round(jaccard, 6),
                "matching_strata_l1_delta": _distribution_l1(left, right),
                "interpretation": "Differences can only come from the id/seed hash tie-breaker.",
            }
        )

    primary_quotas = _distribution(targeted)
    fixed_hash_selections = {
        seed: _select_with_fixed_quotas(candidates, primary_quotas, seed, "hash")
        for seed in seeds
    }
    fixed_hash_rows, fixed_hash_jaccards = _pairwise_jaccard_rows(
        fixed_hash_selections,
        "fixed_quota_hash",
    )
    null_seeds = list(range(20260800, 20260800 + NULL_REPLICATES))
    null_selections = {
        seed: _select_with_fixed_quotas(candidates, primary_quotas, seed, "random")
        for seed in null_seeds
    }
    null_rows, null_jaccards = _pairwise_jaccard_rows(
        null_selections,
        "fixed_quota_stratified_random_null",
    )

    reversed_selection = select_error_guided(
        list(reversed(candidates)),
        profile_rows,
        budget=budget,
        seed=primary_seed,
    )
    shuffled_candidates = list(candidates)
    random.Random(20260901).shuffle(shuffled_candidates)
    shuffled_selection = select_error_guided(
        shuffled_candidates,
        profile_rows,
        budget=budget,
        seed=primary_seed,
    )
    input_order_invariant = (
        targeted_ids == {row["id"] for row in reversed_selection}
        and targeted_ids == {row["id"] for row in shuffled_selection}
    )

    renamed_candidates: list[dict] = []
    renamed_to_original: dict[str, str] = {}
    id_rename_rows: list[dict[str, object]] = []
    for index, row in enumerate(candidates):
        renamed = dict(row)
        renamed_id = f"audit-renamed-{index:05d}"
        renamed["id"] = renamed_id
        renamed_to_original[renamed_id] = row["id"]
        id_rename_rows.append(
            {
                "original_id": row["id"],
                "renamed_id": renamed_id,
                "content_and_metadata_unchanged": True,
            }
        )
        renamed_candidates.append(renamed)
    renamed_selection = select_error_guided(
        renamed_candidates,
        profile_rows,
        budget=budget,
        seed=primary_seed,
    )
    renamed_original_ids = {renamed_to_original[row["id"]] for row in renamed_selection}
    id_rename_jaccard = _jaccard(targeted_ids, renamed_original_ids)

    selected_by_stratum: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in targeted:
        selected_by_stratum[matching_key(row)].add(row["id"])
    rank_rows: list[dict[str, object]] = []
    top_hash_matches = 0
    for key, selected_ids in sorted(selected_by_stratum.items()):
        ordered = sorted(
            by_stratum[key],
            key=lambda row: _stable_noise(row["id"], primary_seed),
            reverse=True,
        )
        top_hash_ids = {row["id"] for row in ordered[: len(selected_ids)]}
        matches = selected_ids == top_hash_ids
        top_hash_matches += int(matches)
        rank_rows.append(
            {
                "task_family": key[0],
                "difficulty_bucket": key[1],
                "answer_magnitude_bucket": key[2],
                "reasoning_length_bucket": key[3],
                "candidate_count": len(by_stratum[key]),
                "selected_count": len(selected_ids),
                "selected_ids_equal_top_hash_k": matches,
            }
        )

    null_mean = mean(null_jaccards)
    fixed_hash_mean = mean(fixed_hash_jaccards) if fixed_hash_jaccards else 1.0
    null_p05 = _percentile(null_jaccards, 0.05)
    null_p95 = _percentile(null_jaccards, 0.95)
    hash_consistent_with_null = null_p05 <= fixed_hash_mean <= null_p95
    robustness_rows: list[dict[str, object]] = [
        {
            "metric": "input_order_invariant",
            "value": input_order_invariant,
            "interpretation": "Reversing and shuffling input order do not change selected IDs.",
        },
        {
            "metric": "candidate_id_rename_jaccard",
            "value": round(id_rename_jaccard, 6),
            "interpretation": "Content is unchanged; only candidate IDs are renamed.",
        },
        {
            "metric": "fixed_quota_hash_mean_jaccard",
            "value": round(fixed_hash_mean, 6),
            "interpretation": "Hash selections use the primary targeted stratum quotas.",
        },
        {
            "metric": "fixed_quota_random_null_mean_jaccard",
            "value": round(null_mean, 6),
            "interpretation": "Null uses stratified random selections with the same quotas.",
        },
        {
            "metric": "fixed_quota_random_null_p05",
            "value": round(null_p05, 6),
            "interpretation": "Fifth percentile of pairwise null Jaccard.",
        },
        {
            "metric": "fixed_quota_random_null_p95",
            "value": round(null_p95, 6),
            "interpretation": "Ninety-fifth percentile of pairwise null Jaccard.",
        },
        {
            "metric": "fixed_quota_hash_consistent_with_random_null",
            "value": hash_consistent_with_null,
            "interpretation": "Hash overlap falls inside the random-null 5-95% interval.",
        },
    ]

    score_fields_fully_matched = set(PROFILE_SIGNAL_FIELDS).issubset(MATCHING_FIELDS)
    has_nonrandom_instance_signal = bool(INSTANCE_LEVEL_SIGNAL_FIELDS)
    identifiable = not score_fields_fully_matched or has_nonrandom_instance_signal
    verdict = "pass" if identifiable else "fail"

    summary: dict[str, object] = {
        "audit_version": AUDIT_VERSION,
        "claim_under_audit": (
            "The current selector contains selection signal beyond the metadata used by "
            "exact matched random."
        ),
        "verdict": verdict,
        "identifiability_status": (
            "identifiable_beyond_matching_metadata"
            if identifiable
            else "not_identifiable_beyond_matching_metadata"
        ),
        "candidate_count": len(candidates),
        "profile_row_count": len(profile_rows),
        "budget": budget,
        "seeds": seeds,
        "matching_fields": list(MATCHING_FIELDS),
        "profile_signal_fields": list(PROFILE_SIGNAL_FIELDS),
        "instance_level_signal_fields": list(INSTANCE_LEVEL_SIGNAL_FIELDS),
        "tie_breaker_fields": list(TIE_BREAKER_FIELDS),
        "score_fields_fully_matched_by_primary_baseline": score_fields_fully_matched,
        "has_nonrandom_instance_level_signal": has_nonrandom_instance_signal,
        "stratum_count": len(by_stratum),
        "strata_with_multiple_candidates": sum(len(rows) > 1 for rows in by_stratum.values()),
        "max_unique_nonrandom_weight_count_within_stratum": max_weight_variants,
        "profile_covered_candidate_count": covered_candidates,
        "profile_covered_candidate_rate": round(covered_candidates / len(candidates), 6),
        "targeted_matched_strata_l1_delta": _distribution_l1(targeted, matched),
        "targeted_matched_overlap_count": len(targeted_ids & matched_ids),
        "targeted_matched_overlap_rate": round(len(targeted_ids & matched_ids) / budget, 6),
        "mean_targeted_id_jaccard_across_seed_pairs": (
            round(mean(pair_jaccards), 6) if pair_jaccards else 1.0
        ),
        "max_targeted_matching_strata_l1_across_seed_pairs": max(
            (int(row["matching_strata_l1_delta"]) for row in seed_rows),
            default=0,
        ),
        "selector_seed_changes_stratum_quotas": any(
            int(row["matching_strata_l1_delta"]) > 0 for row in seed_rows
        ),
        "input_order_invariant": input_order_invariant,
        "candidate_id_rename_jaccard": round(id_rename_jaccard, 6),
        "within_stratum_top_hash_match_rate": round(
            top_hash_matches / len(rank_rows) if rank_rows else 1.0,
            6,
        ),
        "fixed_quota_hash_mean_jaccard": round(fixed_hash_mean, 6),
        "fixed_quota_random_null_mean_jaccard": round(null_mean, 6),
        "fixed_quota_random_null_p05": round(null_p05, 6),
        "fixed_quota_random_null_p95": round(null_p95, 6),
        "fixed_quota_hash_consistent_with_random_null": hash_consistent_with_null,
        "null_pairwise_jaccard_count": len(null_jaccards),
        "null_pairwise_samples_independent": False,
        "null_interval_interpretation": (
            "Descriptive 5-95% interval over dependent pairwise comparisons; "
            "not an independent-sample significance test."
        ),
        "training_allowed": False,
        "selection_effectiveness_claim_allowed": False,
        "required_fix": (
            "Define a preregistered candidate-level operation/skill representation and score "
            "residual diagnostic errors after coarse metadata is controlled."
        ),
        "failure_reason": (
            "All nonrandom profile score fields are included in the exact matching key. "
            "Within a matched stratum, current ordering is only an id/seed hash tie-breaker."
        ),
    }
    return (
        summary,
        stratum_rows,
        seed_rows,
        fixed_hash_rows + null_rows,
        robustness_rows,
        rank_rows,
        id_rename_rows,
    )


def write_outputs(
    output_dir: Path,
    summary: dict[str, object],
    stratum_rows: list[dict[str, object]],
    seed_rows: list[dict[str, object]],
    null_rows: list[dict[str, object]],
    robustness_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    id_rename_rows: list[dict[str, object]],
    candidate_path: Path,
    profile_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "stratum_audit.csv", stratum_rows)
    write_csv(output_dir / "seed_sensitivity.csv", seed_rows)
    write_csv(output_dir / "null_jaccard.csv", null_rows)
    write_csv(output_dir / "robustness_checks.csv", robustness_rows)
    write_csv(output_dir / "within_stratum_rank.csv", rank_rows)
    write_csv(output_dir / "id_rename_map.csv", id_rename_rows)
    metadata = {
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "model_generation_run": False,
        "training_run": False,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "candidate_sha256": _sha256(candidate_path),
        "profile_path": str(profile_path.relative_to(ROOT)),
        "profile_sha256": _sha256(profile_path),
        "selector_source_path": "src/eg_sft/selection/error_guided.py",
        "selector_source_sha256": _sha256(ROOT / "src/eg_sft/selection/error_guided.py"),
        "matched_random_source_path": "src/eg_sft/selection/matched_random.py",
        "matched_random_source_sha256": _sha256(
            ROOT / "src/eg_sft/selection/matched_random.py"
        ),
        "tie_break_rule": "sha256(f'{seed}:{candidate_id}') first 12 hex digits / 16**12",
        "tie_break_direction": "descending",
        "id_rename_rule": "candidate row index -> audit-renamed-{index:05d}",
        "id_rename_map_path": "results/selector_identifiability_audit/id_rename_map.csv",
        "null_pairwise_samples_independent": False,
        "null_interval_interpretation": "descriptive_only_not_significance_test",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline audit of whether the current selector is identifiable beyond matched metadata."
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "data" / "samples" / "candidate_pool.jsonl",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "results" / "real_error_profile.csv",
    )
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seeds", default="20260710,20260711,20260712")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "selector_identifiability_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate.resolve()
    profile_path = args.profile.resolve()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    (
        summary,
        stratum_rows,
        seed_rows,
        null_rows,
        robustness_rows,
        rank_rows,
        id_rename_rows,
    ) = analyze_selector_identifiability(
        read_jsonl(candidate_path),
        read_csv(profile_path),
        budget=args.budget,
        seeds=seeds,
    )
    write_outputs(
        args.output_dir.resolve(),
        summary,
        stratum_rows,
        seed_rows,
        null_rows,
        robustness_rows,
        rank_rows,
        id_rename_rows,
        candidate_path,
        profile_path,
    )
    print(
        f"selector identifiability verdict={summary['verdict']} "
        f"status={summary['identifiability_status']} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
