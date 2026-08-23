"""Frozen matrix-level comparison for the cloud B=500 replication."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from eg_sft.training.b500 import file_sha256


STRATEGIES = ("random", "rds_all", "rds_error")
SEEDS = (17, 29, 41)
EXPECTED_EXAMPLE_COUNT = 1319
DOWNSTREAM_EFFECT_GATE = 0.015
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260823


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _passing_audit(run_dir: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    audit_paths = sorted((run_dir / "audits").glob("formal_audit*.json"))
    passing: list[tuple[Path, dict[str, Any]]] = []
    for path in audit_paths:
        audit = _read_json(path)
        if audit.get("status") == "PASS" and audit.get("source_run_id") == run_id:
            passing.append((path, audit))
    if not passing:
        raise ValueError(f"{run_dir.name} has no passing formal audit")
    return passing[-1]


def load_completed_run(run_dir: Path) -> dict[str, Any]:
    """Load one formally audited run and recompute its item-level accuracy."""

    manifest_path = run_dir / "manifest.json"
    completion_path = run_dir / "run_complete.json"
    metrics_path = run_dir / "evaluation" / "metrics.json"
    raw_path = run_dir / "evaluation" / "raw_outputs.jsonl"
    for path in (manifest_path, completion_path, metrics_path, raw_path):
        if not path.is_file():
            raise ValueError(f"missing formal artifact: {path}")

    manifest = _read_json(manifest_path)
    completion = _read_json(completion_path)
    metrics = _read_json(metrics_path)
    run_id = str(manifest.get("run_id", ""))
    strategy = str(manifest.get("config", {}).get("strategy", ""))
    seed = int(manifest.get("seed", -1))
    if strategy not in STRATEGIES or seed not in SEEDS:
        raise ValueError(f"unexpected formal job: {strategy}/{seed}")
    if completion.get("status") != "PASS" or completion.get("next_job_started") is not False:
        raise ValueError(f"{run_dir.name} is not a closed passing run")
    if metrics.get("strategy") != strategy or int(metrics.get("seed", -1)) != seed:
        raise ValueError(f"{run_dir.name} metrics identify a different job")

    raw_sha256 = file_sha256(raw_path)
    metrics_sha256 = file_sha256(metrics_path)
    if raw_sha256 != metrics.get("raw_outputs_sha256"):
        raise ValueError(f"{run_dir.name} raw output hash differs from metrics")
    if raw_sha256 != completion.get("raw_outputs_sha256"):
        raise ValueError(f"{run_dir.name} raw output hash differs from completion")
    if metrics_sha256 != completion.get("evaluation_metrics_sha256"):
        raise ValueError(f"{run_dir.name} metrics hash differs from completion")

    audit_path, audit = _passing_audit(run_dir, run_id)
    if audit.get("strategy") != strategy or int(audit.get("seed", -1)) != seed:
        raise ValueError(f"{run_dir.name} passing audit identifies a different job")

    rows = _read_jsonl(raw_path)
    expected_count = int(metrics.get("example_count", -1))
    if expected_count != EXPECTED_EXAMPLE_COUNT or len(rows) != EXPECTED_EXAMPLE_COUNT:
        raise ValueError(f"{run_dir.name} must contain {EXPECTED_EXAMPLE_COUNT} outputs")
    correct_by_id: dict[str, bool] = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        numeric_correct = row.get("numeric_correct")
        if not record_id or not isinstance(numeric_correct, bool):
            raise ValueError(f"{run_dir.name} has an invalid item-level output")
        if record_id in correct_by_id:
            raise ValueError(f"{run_dir.name} has duplicate record_id {record_id}")
        correct_by_id[record_id] = numeric_correct

    correct_count = sum(correct_by_id.values())
    accuracy = correct_count / EXPECTED_EXAMPLE_COUNT
    if correct_count != int(metrics.get("numeric_correct_count", -1)):
        raise ValueError(f"{run_dir.name} numeric correct count cannot be reproduced")
    if not math.isclose(accuracy, float(metrics.get("numeric_accuracy", -1.0)), abs_tol=1e-15):
        raise ValueError(f"{run_dir.name} numeric accuracy cannot be reproduced")

    return {
        "run_id": run_id,
        "run_dir": run_dir.name,
        "strategy": strategy,
        "seed": seed,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "strict_parse_rate": float(metrics["strict_parse_rate"]),
        "correct_by_id": correct_by_id,
        "hashes": {
            "raw_outputs_sha256": raw_sha256,
            "evaluation_metrics_sha256": metrics_sha256,
            "formal_audit_sha256": file_sha256(audit_path),
        },
    }


def load_complete_matrix(run_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Require exactly one passing run for every strategy-by-seed cell."""

    expected = {(strategy, seed) for strategy in STRATEGIES for seed in SEEDS}
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    if not run_root.is_dir():
        raise ValueError(f"run root does not exist: {run_root}")
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        if not (run_dir / "manifest.json").is_file():
            continue
        run = load_completed_run(run_dir)
        key = (run["strategy"], run["seed"])
        if key in loaded:
            raise ValueError(f"duplicate formal matrix cell: {key}")
        loaded[key] = run
    missing = sorted(expected - set(loaded))
    extra = sorted(set(loaded) - expected)
    if missing or extra:
        raise ValueError(f"formal matrix is incomplete or invalid: missing={missing}, extra={extra}")

    reference_ids = set(loaded[(STRATEGIES[0], SEEDS[0])]["correct_by_id"])
    for key, run in loaded.items():
        if set(run["correct_by_id"]) != reference_ids:
            raise ValueError(f"record IDs differ for matrix cell {key}")
    return loaded


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def hierarchical_bootstrap_error_minus_all(
    matrix: dict[tuple[str, int], dict[str, Any]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample seeds, then paired GSM8K items within each sampled seed."""

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = random.Random(bootstrap_seed)
    record_ids = sorted(matrix[("rds_all", SEEDS[0])]["correct_by_id"])
    differences: list[float] = []
    for _ in range(replicates):
        total = 0
        count = 0
        for _ in SEEDS:
            seed = rng.choice(SEEDS)
            error_rows = matrix[("rds_error", seed)]["correct_by_id"]
            all_rows = matrix[("rds_all", seed)]["correct_by_id"]
            for _ in record_ids:
                record_id = rng.choice(record_ids)
                total += int(error_rows[record_id]) - int(all_rows[record_id])
                count += 1
        differences.append(total / count)
    differences.sort()
    return {
        "definition": "rds_error numeric accuracy minus rds_all numeric accuracy",
        "unit": "accuracy_fraction",
        "replicates": replicates,
        "seed": bootstrap_seed,
        "ci_95": [
            _quantile(differences, 0.025),
            _quantile(differences, 0.975),
        ],
        "probability_difference_above_zero": sum(value > 0 for value in differences)
        / replicates,
    }


def analyze_complete_matrix(
    matrix: dict[tuple[str, int], dict[str, Any]],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute the preregistered B=500 comparison without changing its gate."""

    per_job = []
    for seed in SEEDS:
        for strategy in STRATEGIES:
            run = matrix[(strategy, seed)]
            per_job.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "accuracy": run["accuracy"],
                    "correct_count": run["correct_count"],
                    "strict_parse_rate": run["strict_parse_rate"],
                    "run_id": run["run_id"],
                    "run_dir": run["run_dir"],
                    "hashes": run["hashes"],
                }
            )

    strategy_summary: dict[str, Any] = {}
    for strategy in STRATEGIES:
        accuracies = [matrix[(strategy, seed)]["accuracy"] for seed in SEEDS]
        strategy_summary[strategy] = {
            "mean_accuracy": mean(accuracies),
            "sample_standard_deviation": stdev(accuracies),
            "seed_accuracies": {str(seed): matrix[(strategy, seed)]["accuracy"] for seed in SEEDS},
        }

    paired_seed_differences: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        random_accuracy = matrix[("random", seed)]["accuracy"]
        all_accuracy = matrix[("rds_all", seed)]["accuracy"]
        error_accuracy = matrix[("rds_error", seed)]["accuracy"]
        paired_seed_differences[str(seed)] = {
            "rds_error_minus_rds_all": error_accuracy - all_accuracy,
            "rds_error_minus_random": error_accuracy - random_accuracy,
            "rds_all_minus_random": all_accuracy - random_accuracy,
        }

    error_minus_all = [
        paired_seed_differences[str(seed)]["rds_error_minus_rds_all"] for seed in SEEDS
    ]
    mean_error_minus_all = mean(error_minus_all)
    positive_seed_count = sum(value > 0 for value in error_minus_all)
    gate_passed = (
        mean_error_minus_all >= DOWNSTREAM_EFFECT_GATE and positive_seed_count >= 2
    )

    wins = losses = ties = 0
    for seed in SEEDS:
        error_rows = matrix[("rds_error", seed)]["correct_by_id"]
        all_rows = matrix[("rds_all", seed)]["correct_by_id"]
        for record_id in sorted(error_rows):
            difference = int(error_rows[record_id]) - int(all_rows[record_id])
            if difference > 0:
                wins += 1
            elif difference < 0:
                losses += 1
            else:
                ties += 1

    bootstrap = hierarchical_bootstrap_error_minus_all(
        matrix,
        replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    if gate_passed:
        boundary = (
            "The cloud B=500 matrix meets the frozen downstream escalation gate in this "
            "model, dataset and training-budget setting. It does not establish generality "
            "across models, datasets or budgets."
        )
    else:
        boundary = (
            "The cloud B=500 matrix does not meet the frozen downstream escalation gate. "
            "This is not proof that the selector has zero effect in every setting."
        )
    return {
        "analysis_schema_version": "b500-cloud-matrix-analysis-v1",
        "matrix_complete": True,
        "strategies": list(STRATEGIES),
        "seeds": list(SEEDS),
        "example_count_per_job": EXPECTED_EXAMPLE_COUNT,
        "per_job": per_job,
        "strategy_summary": strategy_summary,
        "paired_seed_differences": paired_seed_differences,
        "frozen_downstream_gate": {
            "definition": "rds_error minus rds_all",
            "minimum_mean_accuracy_difference": DOWNSTREAM_EFFECT_GATE,
            "minimum_positive_paired_seeds": 2,
            "observed_mean_accuracy_difference": mean_error_minus_all,
            "observed_positive_paired_seeds": positive_seed_count,
            "passed": gate_passed,
        },
        "item_by_seed_direction_counts": {
            "rds_error_wins": wins,
            "rds_error_losses": losses,
            "ties": ties,
        },
        "hierarchical_bootstrap": bootstrap,
        "claim_boundary": boundary,
    }
