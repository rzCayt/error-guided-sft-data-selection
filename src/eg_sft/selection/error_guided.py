from __future__ import annotations

import hashlib
from collections import defaultdict


def _stable_noise(identifier: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def build_profile_weights(profile_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    weights: dict[tuple[str, str, str, str], float] = defaultdict(float)
    for row in profile_rows:
        key = (
            row["task_family"],
            row["difficulty_bucket"],
            row["answer_magnitude_bucket"],
            row["reasoning_length_bucket"],
        )
        total = float(row.get("count", 0) or 0)
        failures = float(row.get("failures", 0) or 0)
        error_rate = failures / total if total else 0.0
        weights[key] += 1.0 + 4.0 * error_rate
    return dict(weights)


def select_error_guided(
    candidates: list[dict],
    profile_rows: list[dict[str, str]],
    budget: int,
    seed: int = 20260707,
) -> list[dict]:
    weights = build_profile_weights(profile_rows)

    def score(row: dict) -> tuple[float, float]:
        buckets = row["buckets"]
        key = (
            row["task_family"],
            buckets["difficulty_bucket"],
            buckets["answer_magnitude_bucket"],
            buckets["reasoning_length_bucket"],
        )
        return (weights.get(key, 1.0), _stable_noise(row["id"], seed))

    ordered = sorted(candidates, key=score, reverse=True)
    return ordered[:budget]
