#!/usr/bin/env python3
"""Analyze the unified 48x3 fixed-state U0a reliability experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from eg_sft.analysis.state_dependence import bootstrap_interval, u0_point_metrics
from eg_sft.experiment.utility import icc_absolute_agreement


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


def analyze(
    *, protocol_path: Path, panel_path: Path, run_dir: Path, output_root: Path
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    protocol = read_json(protocol_path)
    panel = read_json(panel_path)
    audit = read_json(run_dir / "INDEPENDENT_AUDIT.json")
    if audit.get("schema_version") != "candidate-utility-state-probe-audit-v3":
        raise ValueError("U0 run does not have a v3 independent audit")
    if audit.get("status") != "PASS" or audit.get("state_id") != "zero_initialized_lora":
        raise ValueError("U0 v3 run did not pass independent audit")
    if audit.get("historical_measurements_reused") is not False:
        raise ValueError("U0 v3 run reused historical measurements")

    candidate_ids = [str(row["candidate_id"]) for row in panel["candidates"]]
    stage = protocol["stage_u0a_fixed_state_reliability"]
    seeds = [int(seed) for seed in stage["probe_seeds"]]
    rows = read_jsonl(run_dir / "utility_measurements.jsonl")
    values: dict[str, dict[int, float]] = {candidate_id: {} for candidate_id in candidate_ids}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        seed = int(row["probe_seed"])
        if candidate_id not in values or seed not in seeds:
            raise ValueError(f"measurement outside frozen U0 matrix: {candidate_id}/seed{seed}")
        if seed in values[candidate_id]:
            raise ValueError(f"duplicate U0 measurement: {candidate_id}/seed{seed}")
        values[candidate_id][seed] = float(row["utility"])
    if any(set(seed_values) != set(seeds) for seed_values in values.values()):
        raise ValueError("U0 v3 matrix is incomplete")
    if len(rows) != int(stage["new_measurements"]):
        raise ValueError("U0 v3 row count differs from protocol")

    matrix = [[values[candidate_id][seed] for seed in seeds] for candidate_id in candidate_ids]
    point = u0_point_metrics(matrix, seeds)
    repetitions = int(stage["bootstrap_repetitions"])
    icc_interval = bootstrap_interval(
        sample_size=len(candidate_ids),
        statistic=lambda indices: icc_absolute_agreement([matrix[index] for index in indices]),
        repetitions=repetitions,
        seed=2026083101,
    )
    gate = stage["go_gate"]
    go = (
        point["icc_absolute_agreement_a1"] >= float(gate["minimum_icc_a1_point"])
        and float(icc_interval["lower_95"]) >= float(gate["minimum_icc_a1_ci_lower"])
        and point["median_pairwise_spearman"]
        >= float(gate["minimum_median_pairwise_spearman"])
        and point["minimum_pairwise_spearman"]
        >= float(gate["minimum_pairwise_spearman"])
    )
    stop = (
        point["icc_absolute_agreement_a1"] < float(stage["stop_gate"]["icc_a1_below"])
        or point["minimum_pairwise_spearman"]
        < float(stage["stop_gate"]["any_pairwise_spearman_below"])
    )
    verdict = "GO" if go else "STOP" if stop else "AMBIGUOUS"

    output_root.mkdir(parents=True, exist_ok=False)
    epsilon = float(point["median_within_candidate_sd"])
    with (output_root / "candidate_seed_utilities.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for candidate_id, row_values in values.items():
            ordered = [row_values[seed] for seed in seeds]
            handle.write(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "utility_by_probe_seed": {
                            str(seed): row_values[seed] for seed in seeds
                        },
                        "utility_mean": statistics.mean(ordered),
                        "utility_sd": statistics.stdev(ordered),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    result = {
        "schema_version": "candidate-utility-u0a-fixed-state-reliability-v3",
        "status": verdict,
        "candidate_count": len(candidate_ids),
        "probe_seeds": seeds,
        "historical_measurements_reused": False,
        **point,
        "icc_bootstrap_95": icc_interval,
        "practical_sign_epsilon": epsilon,
        "gate": gate,
        "next_action": {
            "GO": "RUN_FROZEN_U1_INITIAL_FOUR_STATES_TWO_PROBE_SEEDS",
            "AMBIGUOUS": "ADD_TWO_FROZEN_PROBE_SEEDS_THEN_REASSESS_ONCE",
            "STOP": "STOP_CROSS_STATE_CLAIMS_AND_STUDY_MEASUREMENT_UNCERTAINTY",
        }[verdict],
    }
    (output_root / "U0A_VERDICT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        protocol_path=args.protocol.resolve(),
        panel_path=args.panel.resolve(),
        run_dir=args.run_dir.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps({"status": result["status"], "candidate_count": result["candidate_count"]}))


if __name__ == "__main__":
    main()
