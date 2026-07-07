from __future__ import annotations

import random
from collections import defaultdict


def matching_key(row: dict) -> tuple[str, str, str, str]:
    buckets = row["buckets"]
    return (
        row["task_family"],
        buckets["difficulty_bucket"],
        buckets["answer_magnitude_bucket"],
        buckets["reasoning_length_bucket"],
    )


def select_matched_random(
    candidates: list[dict],
    target_rows: list[dict],
    seed: int = 20260708,
) -> list[dict]:
    rng = random.Random(seed)
    by_key: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    target_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    target_ids = {row["id"] for row in target_rows}

    for row in candidates:
        if row["id"] not in target_ids:
            by_key[matching_key(row)].append(row)
    for row in target_rows:
        target_counts[matching_key(row)] += 1

    selected: list[dict] = []
    for key, count in sorted(target_counts.items()):
        pool = by_key.get(key, [])
        rng.shuffle(pool)
        selected.extend(pool[:count])

    if len(selected) < len(target_rows):
        selected_ids = {row["id"] for row in selected} | target_ids
        remainder = [row for row in candidates if row["id"] not in selected_ids]
        rng.shuffle(remainder)
        selected.extend(remainder[: len(target_rows) - len(selected)])

    return selected[: len(target_rows)]
