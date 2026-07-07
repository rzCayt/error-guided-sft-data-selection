from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict

from eg_sft.selection.bias_audit import summarize_selection_bias
from eg_sft.selection.matched_random import matching_key, select_matched_random

DIFFICULTY_SCORE = {"easy": 1.0, "medium": 2.0, "hard": 3.0}


def _stable_noise(identifier: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _allocate_proportional_counts(
    counts: dict[tuple[str, str, str, str], int],
    budget: int,
) -> dict[tuple[str, str, str, str], int]:
    total = sum(counts.values())
    if total <= 0 or budget <= 0:
        return {}

    raw = {key: budget * count / total for key, count in counts.items()}
    allocated = {key: min(counts[key], int(value)) for key, value in raw.items()}
    remaining = budget - sum(allocated.values())
    by_fraction = sorted(
        counts,
        key=lambda key: (raw[key] - int(raw[key]), counts[key], key),
        reverse=True,
    )

    while remaining > 0:
        progressed = False
        for key in by_fraction:
            if allocated[key] >= counts[key]:
                continue
            allocated[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return allocated


def select_exact_matched_random_multi_seed(
    candidates: list[dict],
    target_rows: list[dict],
    seeds: list[int],
) -> dict[int, list[dict]]:
    return {
        seed: select_matched_random(candidates, target_rows, seed=seed)
        for seed in seeds
    }


def select_stratified_random(
    candidates: list[dict],
    budget: int,
    seed: int = 20260711,
) -> list[dict]:
    rng = random.Random(seed)
    by_key: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        by_key[matching_key(row)].append(row)

    quotas = _allocate_proportional_counts(
        {key: len(rows) for key, rows in by_key.items()},
        budget,
    )
    selected: list[dict] = []
    for key, quota in sorted(quotas.items()):
        pool = list(by_key[key])
        rng.shuffle(pool)
        selected.extend(pool[:quota])

    if len(selected) < budget:
        selected_ids = {row["id"] for row in selected}
        remainder = [row for row in candidates if row["id"] not in selected_ids]
        rng.shuffle(remainder)
        selected.extend(remainder[: budget - len(selected)])
    return selected[:budget]


def _profile_error_rates(profile_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    rates: dict[tuple[str, str, str, str], float] = {}
    for row in profile_rows:
        key = (
            row["task_family"],
            row["difficulty_bucket"],
            row["answer_magnitude_bucket"],
            row["reasoning_length_bucket"],
        )
        count = float(row.get("count", 0) or 0)
        failures = float(row.get("failures", 0) or 0)
        rates[key] = failures / count if count else 0.0
    return rates


def select_metadata_hard_baseline(
    candidates: list[dict],
    profile_rows: list[dict[str, str]],
    budget: int,
    seed: int = 20260712,
) -> list[dict]:
    error_rates = _profile_error_rates(profile_rows)

    def score(row: dict) -> tuple[float, float, float]:
        key = matching_key(row)
        difficulty = DIFFICULTY_SCORE.get(row["buckets"]["difficulty_bucket"], 0.0)
        error_rate = error_rates.get(key, 0.0)
        return (error_rate, difficulty, _stable_noise(row["id"], seed))

    ordered = sorted(candidates, key=score, reverse=True)
    return ordered[:budget]


def summarize_baseline_suite(
    targeted: list[dict],
    baselines: dict[str, list[dict]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    matched_seed_metrics: dict[str, list[float]] = defaultdict(list)
    target_dist = Counter(matching_key(row) for row in targeted)
    for name, baseline_rows in sorted(baselines.items()):
        baseline_dist = Counter(matching_key(row) for row in baseline_rows)
        strata_l1 = sum(
            abs(target_dist.get(key, 0) - baseline_dist.get(key, 0))
            for key in set(target_dist) | set(baseline_dist)
        )
        for metric in summarize_selection_bias(targeted, baseline_rows):
            metric_name = str(metric["metric"])
            note = str(metric["note"])
            if metric_name == "matched_random_count":
                metric_name = "baseline_count"
                note = "Number of selected baseline examples."
            if metric_name in {"overlap_count", "overlap_rate"} and not name.startswith(
                "exact_matched_random_seed_"
            ):
                note = "Shared examples between targeted and this auxiliary baseline."
            if name.startswith("exact_matched_random_seed_"):
                value = metric["matched_random"]
                if isinstance(value, (int, float)):
                    matched_seed_metrics[f"{metric_name}_baseline_value"].append(float(value))
                delta = metric["delta"]
                if isinstance(delta, (int, float)):
                    matched_seed_metrics[f"{metric_name}_delta"].append(float(delta))
            rows.append(
                {
                    "baseline": name,
                    "metric": metric_name,
                    "targeted": metric["targeted"],
                    "baseline_value": metric["matched_random"],
                    "delta": metric["delta"],
                    "note": note,
                }
            )
        rows.append(
            {
                "baseline": name,
                "metric": "strata_l1_delta",
                "targeted": "",
                "baseline_value": "",
                "delta": strata_l1,
                "note": "L1 count distance between targeted and baseline matching strata.",
            }
        )
        if name.startswith("exact_matched_random_seed_"):
            matched_seed_metrics["strata_l1_delta_delta"].append(float(strata_l1))
    for metric_name, values in sorted(matched_seed_metrics.items()):
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        rows.append(
            {
                "baseline": "exact_matched_random_multi_seed",
                "metric": f"{metric_name}_variance",
                "targeted": "",
                "baseline_value": round(variance, 6),
                "delta": "",
                "note": "Population variance across exact matched-random seeds.",
            }
        )
    return rows
