"""Deterministic source-stratified sampling for H1a candidates."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def _priority(candidate_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest()


def stratified_candidate_sample(
    candidates: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Round-robin source datasets after stable within-source shuffling."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(candidates):
        raise ValueError("count exceeds candidate pool")
    ids = [str(row.get("candidate_id", "")) for row in candidates]
    if any(not candidate_id for candidate_id in ids):
        raise ValueError("every candidate needs candidate_id")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[str(row.get("source_dataset", "unknown"))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: _priority(str(row["candidate_id"]), seed))

    group_names = sorted(
        groups,
        key=lambda name: _priority(name, seed),
    )
    cursors = {name: 0 for name in group_names}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        made_progress = False
        for name in group_names:
            cursor = cursors[name]
            if cursor >= len(groups[name]):
                continue
            selected.append(groups[name][cursor])
            cursors[name] += 1
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise AssertionError("candidate sampling stopped before reaching count")
    return selected
