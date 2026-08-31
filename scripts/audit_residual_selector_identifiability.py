from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.utils.io import read_jsonl, write_csv  # noqa: E402
from eg_sft.eval.metrics import numeric_equal  # noqa: E402
from eg_sft.eval.parser import parse_numeric_strict_final_answer_v3  # noqa: E402

AUDIT_VERSION = "residual_selector_identifiability_v1"
MATCHING_FIELDS = (
    "task_family",
    "difficulty_bucket",
    "answer_magnitude_bucket",
    "reasoning_length_bucket",
)
SIGNAL_VERSION = "contrastive_operation_structure_residual_v1"
CONTROL_VERSION = "bm25_prompt_contrastive_residual_v1"


def matching_key(row: dict) -> tuple[str, str, str, str]:
    buckets = row["buckets"]
    return (
        row["task_family"],
        buckets["difficulty_bucket"],
        buckets["answer_magnitude_bucket"],
        buckets["reasoning_length_bucket"],
    )


def _band(value: float, cuts: tuple[float, float]) -> str:
    if value <= cuts[0]:
        return "low"
    if value <= cuts[1]:
        return "mid"
    return "high"


def operation_features(row: dict) -> frozenset[str]:
    """Extract preregistered operation features without reading answer or rationale."""
    family = row["task_family"]
    params = row.get("metadata", {}).get("params", {})
    features = {f"family={family}"}

    if family == "ratio_change":
        base = int(params["base"])
        pct = int(params["pct"])
        features.update(
            {
                f"direction={params['direction']}",
                f"pct_band={_band(pct, (10, 25))}",
                f"base_digits={len(str(abs(base)))}",
                f"pct_multiple_5={pct % 5 == 0}",
                f"percent_product_integer={(base * pct) % 100 == 0}",
            }
        )
    elif family == "multiplicative_relation":
        factors = [int(value) for value in params["factors"]]
        product = math.prod(factors)
        features.update(
            {
                f"factor_count={len(factors)}",
                f"max_factor_digits={max(len(str(abs(value))) for value in factors)}",
                f"mixed_digit_width={len({len(str(abs(value))) for value in factors}) > 1}",
                f"contains_factor_ge_10={any(value >= 10 for value in factors)}",
                f"product_digits={len(str(abs(product)))}",
            }
        )
    elif family == "weighted_aggregation":
        values = [float(value) for value in params["values"]]
        weights = [float(value) for value in params["weights"]]
        decimal_places = [len(str(value).split(".", 1)[1]) if "." in str(value) else 0 for value in weights]
        features.update(
            {
                f"term_count={len(values)}",
                f"equal_weights={max(weights) - min(weights) < 1e-9}",
                f"max_weight_decimals={max(decimal_places)}",
                f"weight_asymmetry_band={_band(max(weights) - min(weights), (0.1, 0.4))}",
                f"all_terms_integer={all(abs(v * w - round(v * w)) < 1e-9 for v, w in zip(values, weights))}",
                "requires_final_sum=True",
            }
        )
    elif family == "temporal_numeric_constraint":
        deltas = [int(value) for value in params["deltas"]]
        signs = "".join("+" if value >= 0 else "-" for value in deltas)
        features.update(
            {
                f"step_count={len(deltas)}",
                f"sign_pattern={signs}",
                f"mixed_signs={len(set(signs)) > 1}",
                f"has_floor={params.get('floor') is not None}",
                f"has_cap={params.get('cap') is not None}",
                f"max_delta_band={_band(max(abs(value) for value in deltas), (10, 25))}",
            }
        )
    else:
        raise ValueError(f"unsupported task family: {family}")
    return frozenset(features)


def lexical_features(row: dict) -> frozenset[str]:
    normalized = re.sub(r"[-+]?\d+(?:\.\d+)?", " <num> ", row["prompt"].lower())
    return frozenset(re.findall(r"[a-z]+|<num>", normalized))


def _prompt_tokens(row: dict) -> list[str]:
    return re.findall(r"[a-z]+|[-+]?\d+(?:\.\d+)?", row["prompt"].lower())


def bm25_contrastive_scores(
    candidates: list[dict], diagnostic_rows: list[dict]
) -> tuple[dict[str, float], dict[str, bool]]:
    documents = [row for row in diagnostic_rows if row.get("signal_outcome_eligible", True)]
    tokenized = [_prompt_tokens(row) for row in documents]
    document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
    average_length = mean(len(tokens) for tokens in tokenized) if tokenized else 1.0
    n_documents = len(documents)

    def similarity(query: list[str], document: list[str]) -> float:
        frequencies = Counter(document)
        score = 0.0
        for token in set(query):
            frequency = frequencies[token]
            if not frequency:
                continue
            df = document_frequency[token]
            idf = math.log(1.0 + (n_documents - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * len(document) / average_length
            )
            score += idf * frequency * 2.5 / denominator
        return score

    scores: dict[str, float] = {}
    eligible: dict[str, bool] = {}
    for candidate in candidates:
        family_docs = [
            (row, tokens)
            for row, tokens in zip(documents, tokenized)
            if row["task_family"] == candidate["task_family"]
        ]
        failed = [tokens for row, tokens in family_docs if not row["numeric_accuracy"]]
        correct = [tokens for row, tokens in family_docs if row["numeric_accuracy"]]
        eligible[candidate["id"]] = bool(failed and correct)
        if not eligible[candidate["id"]]:
            scores[candidate["id"]] = 0.0
            continue
        query = _prompt_tokens(candidate)
        scores[candidate["id"]] = _top_k_mean(
            [similarity(query, document) for document in failed]
        ) - _top_k_mean([similarity(query, document) for document in correct])
    return scores, eligible


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _top_k_mean(values: list[float], k: int = 3) -> float:
    if not values:
        return 0.0
    return mean(sorted(values, reverse=True)[: min(k, len(values))])


def contrastive_scores(
    candidates: list[dict],
    diagnostic_rows: list[dict],
    feature_fn,
) -> tuple[dict[str, float], dict[str, bool]]:
    by_family_status: dict[tuple[str, bool], list[frozenset[str]]] = defaultdict(list)
    for row in diagnostic_rows:
        if not row.get("signal_outcome_eligible", True):
            continue
        by_family_status[(row["task_family"], bool(row["numeric_accuracy"]))].append(
            feature_fn(row)
        )

    scores: dict[str, float] = {}
    eligible: dict[str, bool] = {}
    for row in candidates:
        family = row["task_family"]
        failed = by_family_status[(family, False)]
        correct = by_family_status[(family, True)]
        eligible[row["id"]] = bool(failed and correct)
        if not eligible[row["id"]]:
            scores[row["id"]] = 0.0
            continue
        features = feature_fn(row)
        failure_similarity = _top_k_mean([_jaccard(features, item) for item in failed])
        correct_similarity = _top_k_mean([_jaccard(features, item) for item in correct])
        scores[row["id"]] = failure_similarity - correct_similarity
    return scores, eligible


def residualize(candidates: list[dict], scores: dict[str, float]) -> dict[str, float]:
    by_stratum: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in candidates:
        by_stratum[matching_key(row)].append(scores[row["id"]])
    centers = {key: mean(values) for key, values in by_stratum.items()}
    return {row["id"]: scores[row["id"]] - centers[matching_key(row)] for row in candidates}


def proportional_quotas(candidates: list[dict], budget: int) -> dict[tuple[str, str, str, str], int]:
    counts = Counter(matching_key(row) for row in candidates)
    raw = {key: budget * count / len(candidates) for key, count in counts.items()}
    quotas = {key: min(counts[key], int(value)) for key, value in raw.items()}
    remaining = budget - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - int(raw[key])), key))
    for key in order:
        if remaining <= 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("unable to allocate fixed quotas")
    return quotas


def _content_key(row: dict) -> str:
    payload = {
        "task_family": row["task_family"],
        "prompt": row["prompt"],
        "metadata": row.get("metadata", {}),
        "buckets": row["buckets"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def select_fixed_quota(
    candidates: list[dict],
    scores: dict[str, float],
    quotas: dict[tuple[str, str, str, str], int],
) -> list[dict]:
    by_stratum: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        by_stratum[matching_key(row)].append(row)
    selected: list[dict] = []
    for key, quota in sorted(quotas.items()):
        ordered = sorted(
            by_stratum[key],
            key=lambda row: (scores[row["id"]], _content_key(row)),
            reverse=True,
        )
        selected.extend(ordered[:quota])
    return selected


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def _jaccard_ids(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def _random_selection(
    candidates: list[dict],
    quotas: dict[tuple[str, str, str, str], int],
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        by_stratum[matching_key(row)].append(row)
    selected: set[str] = set()
    for key, quota in quotas.items():
        selected.update(row["id"] for row in rng.sample(by_stratum[key], quota))
    return selected


def _bootstrap_diagnostics(diagnostic_rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for row in diagnostic_rows:
        groups[(row["task_family"], bool(row["numeric_accuracy"]))].append(row)
    sampled: list[dict] = []
    for rows in groups.values():
        sampled.extend(rng.choice(rows) for _ in range(len(rows)))
    return sampled


def analyze(
    candidates: list[dict],
    diagnostic_outputs: list[dict],
    budget: int = 128,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    if not 0 < budget <= len(candidates):
        raise ValueError("budget must be positive and no larger than candidate count")
    if any(row.get("split") != "candidate_pool" for row in candidates):
        raise ValueError("candidate input must contain candidate_pool only")
    if any(row.get("split") != "dev_diagnostic" for row in diagnostic_outputs):
        raise ValueError("diagnostic input must contain dev_diagnostic only")

    main_raw, main_eligible = contrastive_scores(candidates, diagnostic_outputs, operation_features)
    control_raw, control_eligible = bm25_contrastive_scores(candidates, diagnostic_outputs)
    main = residualize(candidates, main_raw)
    control = residualize(candidates, control_raw)
    quotas = proportional_quotas(candidates, budget)
    main_selected = select_fixed_quota(candidates, main, quotas)
    control_selected = select_fixed_quota(candidates, control, quotas)
    main_ids = {row["id"] for row in main_selected}
    control_ids = {row["id"] for row in control_selected}

    reversed_rows = list(reversed(candidates))
    reversed_main_raw, _ = contrastive_scores(reversed_rows, diagnostic_outputs, operation_features)
    reversed_main = residualize(reversed_rows, reversed_main_raw)
    reversed_ids = {row["id"] for row in select_fixed_quota(reversed_rows, reversed_main, quotas)}

    renamed = [{**row, "id": f"renamed-{index:05d}"} for index, row in enumerate(candidates)]
    renamed_raw, _ = contrastive_scores(renamed, diagnostic_outputs, operation_features)
    renamed_scores = residualize(renamed, renamed_raw)
    renamed_selected = select_fixed_quota(renamed, renamed_scores, quotas)
    renamed_content = {_content_key(row) for row in renamed_selected}
    original_content = {_content_key(row) for row in main_selected}

    random_sets = [_random_selection(candidates, quotas, 20260710 + index) for index in range(30)]
    random_jaccards = [
        _jaccard_ids(left, right) for left, right in itertools.combinations(random_sets, 2)
    ]
    random_mean = mean(random_jaccards)
    bootstrap_sets: list[set[str]] = []
    for index in range(30):
        bootstrap_rows = _bootstrap_diagnostics(diagnostic_outputs, 20260810 + index)
        bootstrap_raw, _ = contrastive_scores(candidates, bootstrap_rows, operation_features)
        bootstrap_scores = residualize(candidates, bootstrap_raw)
        bootstrap_sets.append(
            {
                row["id"]
                for row in select_fixed_quota(candidates, bootstrap_scores, quotas)
            }
        )
    bootstrap_jaccards = [
        _jaccard_ids(left, right) for left, right in itertools.combinations(bootstrap_sets, 2)
    ]
    bootstrap_mean = mean(bootstrap_jaccards)

    eligible_candidate_ids = {
        row["id"] for row in candidates if main_eligible[row["id"]]
    }
    observed_variance = (
        mean(main[identifier] ** 2 for identifier in eligible_candidate_ids)
        if eligible_candidate_ids
        else 0.0
    )
    permutation_rows: list[dict] = []
    permutation_variances: list[float] = []
    permutation_jaccards: list[float] = []
    for index in range(100):
        rng = random.Random(20260910 + index)
        permuted = [dict(row) for row in diagnostic_outputs]
        by_family_indices: dict[str, list[int]] = defaultdict(list)
        for row_index, row in enumerate(permuted):
            if row.get("signal_outcome_eligible", True):
                by_family_indices[row["task_family"]].append(row_index)
        for indices in by_family_indices.values():
            labels = [permuted[row_index]["numeric_accuracy"] for row_index in indices]
            rng.shuffle(labels)
            for row_index, label in zip(indices, labels):
                permuted[row_index]["numeric_accuracy"] = label
        permuted_raw, _ = contrastive_scores(candidates, permuted, operation_features)
        permuted_scores = residualize(candidates, permuted_raw)
        variance = (
            mean(permuted_scores[identifier] ** 2 for identifier in eligible_candidate_ids)
            if eligible_candidate_ids
            else 0.0
        )
        selected_ids = {
            row["id"] for row in select_fixed_quota(candidates, permuted_scores, quotas)
        }
        overlap = _jaccard_ids(main_ids, selected_ids)
        permutation_variances.append(variance)
        permutation_jaccards.append(overlap)
        permutation_rows.append(
            {
                "replicate": index,
                "seed": 20260910 + index,
                "residual_mean_square": round(variance, 10),
                "selected_jaccard_with_observed": round(overlap, 8),
            }
        )
    permutation_tail_rate = (
        1 + sum(value >= observed_variance for value in permutation_variances)
    ) / (1 + len(permutation_variances))

    strata_values: dict[tuple[str, str, str, str], set[float]] = defaultdict(set)
    for row in candidates:
        strata_values[matching_key(row)].add(round(main[row["id"]], 12))
    eligible_strata = {
        matching_key(row) for row in candidates if main_eligible[row["id"]]
    }
    varying_eligible_strata = sum(len(strata_values[key]) > 1 for key in eligible_strata)
    signature_values: dict[tuple[object, ...], set[float]] = defaultdict(set)
    signature_counts: Counter[tuple[object, ...]] = Counter()
    for row in candidates:
        signature = (*matching_key(row), *sorted(operation_features(row)))
        signature_values[signature].add(round(main[row["id"]], 12))
        signature_counts[signature] += 1
    multi_candidate_signatures = [
        key for key, count in signature_counts.items() if count > 1
    ]
    max_variants_within_signature = max(
        (len(signature_values[key]) for key in multi_candidate_signatures), default=0
    )

    score_rows = []
    for row in candidates:
        key = matching_key(row)
        score_rows.append(
            {
                "candidate_id": row["id"],
                "task_family": key[0],
                "difficulty_bucket": key[1],
                "answer_magnitude_bucket": key[2],
                "reasoning_length_bucket": key[3],
                "main_signal_eligible": main_eligible[row["id"]],
                "main_raw_score": round(main_raw[row["id"]], 8),
                "main_residual_score": round(main[row["id"]], 8),
                "control_signal_eligible": control_eligible[row["id"]],
                "control_raw_score": round(control_raw[row["id"]], 8),
                "control_residual_score": round(control[row["id"]], 8),
                "selected_by_main": row["id"] in main_ids,
                "selected_by_control": row["id"] in control_ids,
            }
        )

    ordinal = {"easy": 1.0, "medium": 2.0, "hard": 3.0, "small": 1.0, "large": 3.0, "short": 1.0, "long": 3.0}
    score_vector = [main[row["id"]] for row in candidates]
    confounds = {
        "difficulty_bucket": [ordinal[row["buckets"]["difficulty_bucket"]] for row in candidates],
        "answer_magnitude_bucket": [ordinal.get(row["buckets"]["answer_magnitude_bucket"], 2.0) for row in candidates],
        "reasoning_length_bucket": [ordinal.get(row["buckets"]["reasoning_length_bucket"], 2.0) for row in candidates],
        "prompt_token_count": [float(len(row["prompt"].split())) for row in candidates],
    }
    confound_rows = [
        {
            "signal": SIGNAL_VERSION,
            "confound": name,
            "spearman_rho": round(spearman(score_vector, values), 8),
            "interpretation": "descriptive_only_no_significance_test",
        }
        for name, values in confounds.items()
    ]

    effective_diagnostics = [
        row for row in diagnostic_outputs if row.get("signal_outcome_eligible", True)
    ]
    family_counts = Counter(row["task_family"] for row in effective_diagnostics)
    family_correct = Counter(
        row["task_family"] for row in effective_diagnostics if row["numeric_accuracy"]
    )
    eligible_families = sorted(
        family for family in family_counts if 0 < family_correct[family] < family_counts[family]
    )
    all_families = sorted({row["task_family"] for row in diagnostic_outputs})
    ineligible_families = sorted(set(all_families) - set(eligible_families))
    max_bucket_correlation = max(
        abs(row["spearman_rho"])
        for row in confound_rows
        if row["confound"] in {"difficulty_bucket", "answer_magnitude_bucket", "reasoning_length_bucket"}
    )
    identifiable = (
        bool(eligible_strata)
        and varying_eligible_strata > 0
        and main_ids == reversed_ids
        and original_content == renamed_content
        and max_bucket_correlation < 0.05
        and bootstrap_mean > random_mean
        and max_variants_within_signature > 1
        and permutation_tail_rate <= 0.05
    )
    summary = {
        "audit_version": AUDIT_VERSION,
        "signal_version": SIGNAL_VERSION,
        "control_version": CONTROL_VERSION,
        "verdict": "pass" if identifiable else "fail",
        "identifiability_status": (
            "partially_identifiable_offline"
            if identifiable
            else "not_identifiable_beyond_static_operation_metadata"
        ),
        "candidate_count": len(candidates),
        "diagnostic_count": len(diagnostic_outputs),
        "diagnostic_signal_eligible_count": len(effective_diagnostics),
        "diagnostic_failure_count": sum(not row["numeric_accuracy"] for row in effective_diagnostics),
        "budget": budget,
        "matching_fields": list(MATCHING_FIELDS),
        "eligible_families": eligible_families,
        "ineligible_families": ineligible_families,
        "eligible_stratum_count": len(eligible_strata),
        "varying_eligible_stratum_count": varying_eligible_strata,
        "input_order_invariant": main_ids == reversed_ids,
        "candidate_id_rename_invariant": original_content == renamed_content,
        "main_vs_lexical_selected_jaccard": round(_jaccard_ids(main_ids, control_ids), 8),
        "operation_signature_count": len(signature_counts),
        "multi_candidate_operation_signature_count": len(multi_candidate_signatures),
        "max_unique_score_count_within_operation_signature": max_variants_within_signature,
        "candidate_level_variation_beyond_operation_signature": max_variants_within_signature > 1,
        "main_selection_deterministic_jaccard": 1.0,
        "diagnostic_bootstrap_mean_pairwise_jaccard": round(bootstrap_mean, 8),
        "diagnostic_bootstrap_pairwise_comparisons_independent": False,
        "fixed_quota_random_mean_pairwise_jaccard": round(random_mean, 8),
        "random_pairwise_comparisons_independent": False,
        "observed_residual_mean_square": round(observed_variance, 10),
        "permuted_error_profile_mean_residual_mean_square": round(mean(permutation_variances), 10),
        "permuted_error_profile_tail_rate": round(permutation_tail_rate, 8),
        "permuted_profile_mean_selected_jaccard": round(mean(permutation_jaccards), 8),
        "permutation_interpretation": "exploratory_label_permutation_control_not_preregistered_significance_test",
        "max_absolute_coarse_bucket_spearman": round(max_bucket_correlation, 8),
        "test_split_accessed": False,
        "gold_test_information_used": False,
        "candidate_answer_or_rationale_used": False,
        "training_allowed": False,
        "stability_validation_allowed": False,
        "selection_effectiveness_claim_allowed": False,
        "claim_boundary": (
            "The current failure means the score is identifiable only at a static operation-metadata "
            "level, not at a candidate-specific residual level. It provides no training evidence."
        ),
        "known_blockers": [
            "Some families lack both strict-parser-correct and strict-parser-incorrect examples.",
            "The score is constant after exact stratum plus full operation signature are fixed.",
            "The exploratory observed-profile variance does not clear the 0.05 permutation gate.",
            "Only one frozen target-model diagnostic run is available.",
        ],
        "required_next_step": "Create a human-audited step-level error representation or a model-aware candidate signal, then preregister independent target-model diagnostic replicates.",
    }
    invariance_rows = [
        {"check": "input_order_invariance", "passed": main_ids == reversed_ids, "value": _jaccard_ids(main_ids, reversed_ids)},
        {"check": "candidate_id_rename_invariance", "passed": original_content == renamed_content, "value": _jaccard_ids(original_content, renamed_content)},
        {"check": "fixed_quota_deterministic_stability", "passed": True, "value": 1.0},
        {"check": "diagnostic_bootstrap_stability_above_random", "passed": bootstrap_mean > random_mean, "value": round(bootstrap_mean, 8)},
        {"check": "fixed_quota_random_reference", "passed": True, "value": round(random_mean, 8)},
        {"check": "candidate_variation_beyond_operation_signature", "passed": max_variants_within_signature > 1, "value": max_variants_within_signature},
        {"check": "observed_profile_exceeds_permuted_variance", "passed": permutation_tail_rate <= 0.05, "value": round(permutation_tail_rate, 8)},
        {"check": "test_split_not_accessed", "passed": True, "value": True},
        {"check": "candidate_gold_not_used_by_signal", "passed": True, "value": True},
    ]
    return summary, score_rows, confound_rows, invariance_rows, permutation_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(
    output_dir: Path,
    summary: dict,
    score_rows: list[dict],
    confound_rows: list[dict],
    invariance_rows: list[dict],
    permutation_rows: list[dict],
    candidate_path: Path,
    diagnostic_examples_path: Path,
    diagnostic_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "score_distribution.csv", score_rows)
    write_csv(output_dir / "confound_correlation.csv", confound_rows)
    write_csv(output_dir / "invariance_checks.csv", invariance_rows)
    write_csv(output_dir / "permutation_checks.csv", permutation_rows)
    metadata = {
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "model_generation_run": False,
        "training_run": False,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "candidate_sha256": _sha256(candidate_path),
        "diagnostic_examples_path": str(diagnostic_examples_path.relative_to(ROOT)),
        "diagnostic_examples_sha256": _sha256(diagnostic_examples_path),
        "diagnostic_path": str(diagnostic_path.relative_to(ROOT)),
        "diagnostic_sha256": _sha256(diagnostic_path),
        "allowed_splits": ["candidate_pool", "dev_diagnostic"],
        "forbidden_split": "test_*",
        "main_signal": SIGNAL_VERSION,
        "control_signal": CONTROL_VERSION,
        "main_formula": "mean_top3_jaccard_to_failures - mean_top3_jaccard_to_corrects, then subtract exact-stratum mean",
        "tie_breaker": "sha256(canonical prompt + metadata + buckets), independent of candidate id and input order",
        "candidate_fields_read_by_main_signal": ["task_family", "metadata.params", "buckets"],
        "candidate_fields_explicitly_not_read": ["answer", "rationale", "id as signal"],
        "diagnostic_outcome_field": "offline strict parser v3 rescore of frozen Qwen3-1.7B chat-no-think outputs",
        "diagnostic_outcome_eligibility": "strict parser v3 must produce a numeric value; format failures are excluded from numeric-error contrast",
        "measurement_warning": "Only one frozen target-model diagnostic run is available; bootstrap is not an independent replicate.",
        "random_reference": "30 fixed-quota random replicates; dependent pairwise Jaccard is descriptive only",
        "diagnostic_bootstrap": "30 stratified-within-family-and-outcome resamples; dependent pairwise Jaccard is descriptive only",
        "permutation_control": "100 within-family label permutations; exploratory because added after adversarial review",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline identifiability audit for a preregistered residual selector prototype.")
    parser.add_argument("--candidate", type=Path, default=ROOT / "data" / "samples" / "candidate_pool.jsonl")
    parser.add_argument("--diagnostic-examples", type=Path, default=ROOT / "data" / "samples" / "dev_diagnostic.jsonl")
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=ROOT
        / "results"
        / "qwen3_interface_diagnostic"
        / "qwen3_1_7b_chat_no_think_answer_only"
        / "scale_model_diagnostic_outputs.jsonl",
    )
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "residual_selector_identifiability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate.resolve()
    diagnostic_examples_path = args.diagnostic_examples.resolve()
    diagnostic_path = args.diagnostic.resolve()
    examples_by_id = {row["id"]: row for row in read_jsonl(diagnostic_examples_path)}
    diagnostic_rows = []
    for output in read_jsonl(diagnostic_path):
        if output["id"] not in examples_by_id:
            raise ValueError(f"diagnostic output id missing from dev examples: {output['id']}")
        raw_text = output.get("raw_continuation", output.get("prediction", ""))
        parsed, parse_mode = parse_numeric_strict_final_answer_v3(raw_text)
        diagnostic_rows.append(
            {
                **examples_by_id[output["id"]],
                **output,
                "parsed_prediction": parsed,
                "strict_parse_mode": parse_mode,
                "signal_outcome_eligible": parsed is not None,
                "numeric_accuracy": parsed is not None
                and numeric_equal(parsed, float(output["answer"])),
            }
        )
    summary, scores, correlations, invariance, permutation = analyze(
        read_jsonl(candidate_path), diagnostic_rows, budget=args.budget
    )
    write_outputs(
        args.output_dir.resolve(),
        summary,
        scores,
        correlations,
        invariance,
        permutation,
        candidate_path,
        diagnostic_examples_path,
        diagnostic_path,
    )
    print(f"residual selector identifiability verdict={summary['verdict']} status={summary['identifiability_status']}")


if __name__ == "__main__":
    main()
