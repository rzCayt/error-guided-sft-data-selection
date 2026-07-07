from __future__ import annotations

from collections import Counter

from eg_sft.selection.matched_random import matching_key


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
