"""Pure planning and resume validation for candidate-utility state probes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def measurement_key(state_id: str, candidate_id: str, probe_seed: int) -> tuple[str, str, int]:
    return state_id, candidate_id, int(probe_seed)


def build_measurement_plan(
    *,
    state_id: str,
    candidate_ids: Sequence[str],
    probe_seeds: Sequence[int],
    existing_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    if not state_id:
        raise ValueError("state_id must be non-empty")
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be non-empty and unique")
    seeds = [int(seed) for seed in probe_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("probe seeds must be non-empty and unique")
    existing_keys: set[tuple[str, str, int]] = set()
    for row in existing_rows:
        key = measurement_key(
            str(row["state_id"]), str(row["candidate_id"]), int(row["probe_seed"])
        )
        if key in existing_keys:
            raise ValueError(f"duplicate existing measurement: {key}")
        existing_keys.add(key)
    requested = [
        {
            "state_id": state_id,
            "candidate_id": candidate_id,
            "probe_seed": seed,
        }
        for candidate_id in candidate_ids
        for seed in seeds
    ]
    new_rows = [
        row
        for row in requested
        if measurement_key(row["state_id"], row["candidate_id"], row["probe_seed"])
        not in existing_keys
    ]
    return {
        "state_id": state_id,
        "candidate_count": len(candidate_ids),
        "probe_seeds": seeds,
        "requested_measurement_count": len(requested),
        "reused_measurement_count": len(requested) - len(new_rows),
        "new_measurement_count": len(new_rows),
        "new_measurements": new_rows,
    }


def validate_resume_rows(
    *, plan: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> set[tuple[str, str, int]]:
    planned = {
        measurement_key(row["state_id"], row["candidate_id"], row["probe_seed"])
        for row in plan["new_measurements"]
    }
    completed: set[tuple[str, str, int]] = set()
    for row in rows:
        key = measurement_key(
            str(row["state_id"]), str(row["candidate_id"]), int(row["probe_seed"])
        )
        if key not in planned:
            raise ValueError(f"resume row is outside the frozen plan: {key}")
        if key in completed:
            raise ValueError(f"duplicate resume row: {key}")
        if row.get("status") != "PASS":
            raise ValueError(f"resume row is not PASS: {key}")
        completed.add(key)
    return completed
