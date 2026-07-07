from __future__ import annotations

from collections import Counter

from eg_sft.selection.matched_random import matching_key

REASONING_LENGTH_SCORE = {"short": 1.0, "medium": 2.0, "long": 3.0}


def distribution(rows: list[dict]) -> Counter:
    return Counter(matching_key(row) for row in rows)


def audit_selection_bias(targeted: list[dict], random_rows: list[dict]) -> list[dict[str, object]]:
    keys = sorted(set(distribution(targeted)) | set(distribution(random_rows)))
    target_dist = distribution(targeted)
    random_dist = distribution(random_rows)
    audit = []
    for key in keys:
        audit.append(
            {
                "task_family": key[0],
                "difficulty_bucket": key[1],
                "answer_magnitude_bucket": key[2],
                "reasoning_length_bucket": key[3],
                "targeted_count": target_dist.get(key, 0),
                "matched_random_count": random_dist.get(key, 0),
                "count_delta": target_dist.get(key, 0) - random_dist.get(key, 0),
            }
        )
    return audit


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _marginal_delta(targeted: list[dict], random_rows: list[dict], field: str) -> int:
    target_counts = Counter(row[field] for row in targeted)
    random_counts = Counter(row[field] for row in random_rows)
    keys = set(target_counts) | set(random_counts)
    return sum(abs(target_counts.get(key, 0) - random_counts.get(key, 0)) for key in keys)


def summarize_selection_bias(targeted: list[dict], random_rows: list[dict]) -> list[dict[str, object]]:
    targeted_ids = {row["id"] for row in targeted}
    random_ids = {row["id"] for row in random_rows}
    overlap = targeted_ids & random_ids

    targeted_reasoning = [
        REASONING_LENGTH_SCORE.get(row["buckets"]["reasoning_length_bucket"], 0.0)
        for row in targeted
    ]
    random_reasoning = [
        REASONING_LENGTH_SCORE.get(row["buckets"]["reasoning_length_bucket"], 0.0)
        for row in random_rows
    ]

    return [
        {
            "metric": "targeted_count",
            "targeted": len(targeted),
            "matched_random": "",
            "delta": "",
            "note": "Number of selected error-guided examples.",
        },
        {
            "metric": "matched_random_count",
            "targeted": "",
            "matched_random": len(random_rows),
            "delta": "",
            "note": "Number of selected matched-random examples.",
        },
        {
            "metric": "overlap_count",
            "targeted": len(overlap),
            "matched_random": len(overlap),
            "delta": 0,
            "note": "Shared examples; nonzero overlap is allowed only to preserve exact stratum matching.",
        },
        {
            "metric": "overlap_rate",
            "targeted": round(len(overlap) / len(targeted), 6) if targeted else 0.0,
            "matched_random": round(len(overlap) / len(random_rows), 6) if random_rows else 0.0,
            "delta": 0,
            "note": "Fraction of each subset shared with the other subset.",
        },
        {
            "metric": "mean_abs_answer",
            "targeted": _mean([abs(float(row["answer"])) for row in targeted]),
            "matched_random": _mean([abs(float(row["answer"])) for row in random_rows]),
            "delta": round(
                _mean([abs(float(row["answer"])) for row in targeted])
                - _mean([abs(float(row["answer"])) for row in random_rows]),
                6,
            ),
            "note": "Checks whether answer scale differs beyond bucket matching.",
        },
        {
            "metric": "mean_reasoning_length_score",
            "targeted": _mean(targeted_reasoning),
            "matched_random": _mean(random_reasoning),
            "delta": round(_mean(targeted_reasoning) - _mean(random_reasoning), 6),
            "note": "short=1, medium=2, long=3.",
        },
        {
            "metric": "task_family_marginal_l1_delta",
            "targeted": "",
            "matched_random": "",
            "delta": _marginal_delta(targeted, random_rows, "task_family"),
            "note": "Zero means exact task-family marginal matching.",
        },
        {
            "metric": "difficulty_marginal_l1_delta",
            "targeted": "",
            "matched_random": "",
            "delta": _marginal_delta(targeted, random_rows, "difficulty"),
            "note": "Zero means exact difficulty marginal matching.",
        },
    ]
