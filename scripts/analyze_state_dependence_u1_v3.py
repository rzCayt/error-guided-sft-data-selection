#!/usr/bin/env python3
"""Analyze two-seed zero-to-final local-utility transfer for four frozen states."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from eg_sft.analysis.state_dependence import correlation_with_interval, top_k_jaccard
from eg_sft.analysis.behavior_composition import spearman


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains a non-object row")
    return rows


def _matrix(rows: list[dict[str, Any]], candidate_ids: list[str], seeds: list[int]) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {candidate_id: {} for candidate_id in candidate_ids}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        seed = int(row["probe_seed"])
        if candidate_id not in values or seed not in seeds:
            raise ValueError(f"measurement outside frozen matrix: {candidate_id}/seed{seed}")
        if seed in values[candidate_id]:
            raise ValueError(f"duplicate measurement: {candidate_id}/seed{seed}")
        values[candidate_id][seed] = float(row["utility"])
    if any(set(seed_values) != set(seeds) for seed_values in values.values()):
        raise ValueError("measurement matrix is incomplete")
    return values


def analyze(
    *, protocol_path: Path, u0_root: Path, state_run_dirs: list[Path], output_root: Path
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    protocol = read_json(protocol_path)
    u0_verdict = read_json(u0_root / "U0A_VERDICT.json")
    if u0_verdict.get("status") != "GO":
        raise ValueError("U1 is blocked because U0a did not pass")
    stage = protocol["stage_u1_cross_state_transfer"]
    seeds = [int(seed) for seed in stage["probe_seeds"]]
    expected_states = list(stage["initial_adapter_states"])
    if len(state_run_dirs) != len(expected_states):
        raise ValueError("U1 requires exactly four initial state run directories")

    zero_rows = read_jsonl(u0_root / "candidate_seed_utilities.jsonl")
    candidate_ids = [str(row["candidate_id"]) for row in zero_rows]
    zero_by_id = {
        str(row["candidate_id"]): {
            int(seed): float(value)
            for seed, value in row["utility_by_probe_seed"].items()
            if int(seed) in seeds
        }
        for row in zero_rows
    }
    if any(set(values) != set(seeds) for values in zero_by_id.values()):
        raise ValueError("U0 does not contain the matched U1 probe seeds")
    zero_mean = {
        candidate_id: statistics.mean(zero_by_id[candidate_id].values())
        for candidate_id in candidate_ids
    }
    epsilon = float(u0_verdict["practical_sign_epsilon"])
    repetitions = int(stage["bootstrap_repetitions"])

    state_results: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    for state_index, run_dir in enumerate(state_run_dirs):
        audit = read_json(run_dir / "INDEPENDENT_AUDIT.json")
        state_id = str(audit.get("state_id", ""))
        if audit.get("schema_version") != "candidate-utility-state-probe-audit-v3":
            raise ValueError(f"{state_id}: missing v3 independent audit")
        if audit.get("status") != "PASS" or state_id not in expected_states:
            raise ValueError(f"invalid U1 state audit: {state_id}")
        if state_id in seen_states:
            raise ValueError(f"duplicate U1 state: {state_id}")
        seen_states.add(state_id)
        values = _matrix(read_jsonl(run_dir / "utility_measurements.jsonl"), candidate_ids, seeds)
        state_mean = {
            candidate_id: statistics.mean(values[candidate_id].values())
            for candidate_id in candidate_ids
        }
        zero_vector = [zero_mean[candidate_id] for candidate_id in candidate_ids]
        state_vector = [state_mean[candidate_id] for candidate_id in candidate_ids]
        correlation = correlation_with_interval(
            zero_vector,
            state_vector,
            repetitions=repetitions,
            seed=2026083200 + state_index,
        )
        seed_correlations = {
            str(seed): spearman(
                [zero_by_id[candidate_id][seed] for candidate_id in candidate_ids],
                [values[candidate_id][seed] for candidate_id in candidate_ids],
            )
            for seed in seeds
        }
        jaccard = top_k_jaccard(candidate_ids, zero_vector, state_vector, k=12)
        raw_flip = sum(
            (zero_mean[candidate_id] > 0.0) != (state_mean[candidate_id] > 0.0)
            for candidate_id in candidate_ids
        ) / len(candidate_ids)
        eligible = [
            candidate_id
            for candidate_id in candidate_ids
            if abs(zero_mean[candidate_id]) > epsilon
            and abs(state_mean[candidate_id]) > epsilon
        ]
        practical_flip = (
            sum(
                (zero_mean[candidate_id] > 0.0) != (state_mean[candidate_id] > 0.0)
                for candidate_id in eligible
            )
            / len(eligible)
            if eligible
            else 0.0
        )
        state_results.append(
            {
                "state_id": state_id,
                "zero_vs_state_spearman": correlation,
                "matched_probe_seed_spearman": seed_correlations,
                "top12_jaccard": jaccard,
                "raw_sign_flip_fraction": raw_flip,
                "practical_sign_flip_fraction": practical_flip,
                "practical_sign_candidate_count": len(eligible),
            }
        )
    if seen_states != set(expected_states):
        raise ValueError("U1 state set differs from frozen protocol")

    dependence_gate = stage["state_dependence_gate"]
    stability_gate = stage["state_stability_gate"]
    dependence_states = sum(
        row["zero_vs_state_spearman"]["point"] < 0.75
        and row["zero_vs_state_spearman"]["bootstrap_95"]["upper_95"]
        < float(dependence_gate["maximum_state_ci_upper"])
        for row in state_results
    )
    median_jaccard = statistics.median(row["top12_jaccard"] for row in state_results)
    seed_direction_agrees = all(
        all(value < 0.80 for value in row["matched_probe_seed_spearman"].values())
        or all(value >= 0.80 for value in row["matched_probe_seed_spearman"].values())
        for row in state_results
    )
    dependence = (
        dependence_states
        >= int(dependence_gate["minimum_states_with_spearman_below_0_75"])
        and median_jaccard <= float(dependence_gate["maximum_median_top12_jaccard"])
        and seed_direction_agrees
    )
    stable = all(
        row["zero_vs_state_spearman"]["point"]
        >= float(stability_gate["minimum_all_state_spearman"])
        and row["zero_vs_state_spearman"]["bootstrap_95"]["lower_95"]
        >= float(stability_gate["minimum_all_state_ci_lower"])
        and row["top12_jaccard"]
        >= float(stability_gate["minimum_all_state_top12_jaccard"])
        and row["practical_sign_flip_fraction"]
        < float(stability_gate["maximum_all_state_practical_sign_flip"])
        for row in state_results
    )
    verdict = "STATE_DEPENDENCE" if dependence else "STATE_STABILITY" if stable else "AMBIGUOUS"
    output_root.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "candidate-utility-u1-cross-state-transfer-v3",
        "status": verdict,
        "candidate_count": len(candidate_ids),
        "state_count": len(state_results),
        "probe_seeds": seeds,
        "practical_sign_epsilon": epsilon,
        "states_meeting_dependence_threshold": dependence_states,
        "median_top12_jaccard": median_jaccard,
        "matched_probe_seed_direction_agreement": seed_direction_agrees,
        "state_results": sorted(state_results, key=lambda row: row["state_id"]),
        "next_action": {
            "STATE_DEPENDENCE": "ADD_INTERMEDIATE_CHECKPOINTS_AND_TEST_REVALUATION",
            "STATE_STABILITY": "STOP_STATE_DEPENDENCE_AND_FREEZE_MICRO_SET_PROTOCOL",
            "AMBIGUOUS": stage["ambiguous_action"],
        }[verdict],
        "claim_boundary": (
            "This analysis estimates transfer of local one-step utility across fixed parameter "
            "states. It is not optimizer-trajectory attribution and does not by itself explain "
            "the downstream selector result."
        ),
    }
    (output_root / "U1_VERDICT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--u0-root", type=Path, required=True)
    parser.add_argument("--state-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        protocol_path=args.protocol.resolve(),
        u0_root=args.u0_root.resolve(),
        state_run_dirs=[path.resolve() for path in args.state_run_dir],
        output_root=args.output_root.resolve(),
    )
    print(json.dumps({"status": result["status"], "state_count": result["state_count"]}))


if __name__ == "__main__":
    main()
