"""Unblinded Phase 1A statistics after the sealed 16-cell gate opens."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ("gsm8k", "svamp", "asdiv_numeric", "multiarith")
OOD_TASKS = TASKS[1:]
COMPARISONS = {
    "common_mix_rds_minus_random": (
        "rds_error_common_mix",
        "random_common_mix",
    ),
    "free_mix_rds_minus_random": (
        "rds_error_free_mix",
        "random_free_mix",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def require_all_audited(registry: Mapping[str, Any], *, required: int = 16) -> None:
    jobs = list(registry.get("jobs", []))
    audited = int(registry.get("audited_pass_count", -1))
    if len(jobs) != required or audited != required:
        raise ValueError(f"unblinded analysis requires {required}/{required} audited cells")
    invalid = [str(row.get("cell_id")) for row in jobs if row.get("status") != "AUDITED_PASS"]
    if invalid:
        raise ValueError(f"unblinded analysis found non-audited cells: {invalid}")


def validate_task_alignment(cells: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    if not cells:
        raise ValueError("no completed cells supplied")
    reference: dict[str, list[str]] = {}
    for task in TASKS:
        rows = list(cells[0]["tasks"][task])
        ids = [str(row["record_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate record IDs in {task}")
        reference[task] = ids
    for cell in cells[1:]:
        for task in TASKS:
            ids = [str(row["record_id"]) for row in cell["tasks"][task]]
            if ids != reference[task]:
                raise ValueError(f"task membership/order differs across cells: {task}")
    return reference


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("invalid quantile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric_vector(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    if metric == "accuracy":
        return [float(bool(row["numeric_correct"])) for row in rows]
    if metric == "strict_parse_rate":
        return [float(row.get("strict_parse_status") == "ok") for row in rows]
    if metric == "parse_rate":
        return [float(row.get("parse_status") == "ok") for row in rows]
    raise ValueError(f"unsupported row metric: {metric}")


def cell_metrics(
    cell: Mapping[str, Any], *, token_length: Callable[[str], int] | None = None
) -> dict[str, Any]:
    per_task = {}
    for task in TASKS:
        rows = list(cell["tasks"][task])
        parse_vector = metric_vector(rows, "parse_rate")
        lengths = [
            token_length(str(row.get("raw_output", "")))
            if token_length is not None
            else len(str(row.get("raw_output", "")))
            for row in rows
        ]
        per_task[task] = {
            "record_count": len(rows),
            "accuracy": _mean(metric_vector(rows, "accuracy")),
            "strict_parse_rate": _mean(metric_vector(rows, "strict_parse_rate")),
            "parse_rate": _mean(parse_vector),
            "invalid_generation_rate": 1.0 - _mean(parse_vector),
            "mean_generation_length": _mean([float(value) for value in lengths]),
            "generation_length_unit": "tokens" if token_length is not None else "characters",
        }
    per_task["ood_macro"] = {
        metric: _mean([float(per_task[task][metric]) for task in OOD_TASKS])
        for metric in (
            "accuracy",
            "strict_parse_rate",
            "parse_rate",
            "invalid_generation_rate",
            "mean_generation_length",
        )
    }
    return {
        "cell_id": str(cell["cell_id"]),
        "method": str(cell["method"]),
        "replicate_index": int(cell["replicate_index"]),
        "train_seed": int(cell["train_seed"]),
        "tasks": per_task,
        "training": dict(cell.get("training", {})),
    }


def summarize_methods(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in cell_rows})
    output: dict[str, Any] = {}
    for method in methods:
        group = [row for row in cell_rows if row["method"] == method]
        tasks = {}
        for task in (*TASKS, "ood_macro"):
            tasks[task] = {}
            for metric in (
                "accuracy",
                "strict_parse_rate",
                "parse_rate",
                "invalid_generation_rate",
                "mean_generation_length",
            ):
                values = [float(row["tasks"][task][metric]) for row in group]
                tasks[task][metric] = {
                    "mean": _mean(values),
                    "selection_replicate_sd": _sample_sd(values),
                    "values": values,
                }
        output[method] = {
            "selection_replicate_count": len(group),
            "train_seed_count": len({int(row["train_seed"]) for row in group}),
            "replicate_indices": [int(row["replicate_index"]) for row in group],
            "tasks": tasks,
        }
    return output


def _sample_task_mean(
    *,
    cells: Sequence[Mapping[str, Any]],
    task: str,
    metric: str,
    sampled_cells: Sequence[int],
    sampled_items: Sequence[int],
) -> float:
    values = []
    for cell_index in sampled_cells:
        vector = metric_vector(cells[cell_index]["tasks"][task], metric)
        values.append(_mean([vector[item] for item in sampled_items]))
    return _mean(values)


def independent_selection_item_bootstrap(
    *,
    left_cells: Sequence[Mapping[str, Any]],
    right_cells: Sequence[Mapping[str, Any]],
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 10_000:
        raise ValueError("formal Phase 1 bootstrap requires at least 10,000 replicates")
    if len(left_cells) != 4 or len(right_cells) != 4:
        raise ValueError("Phase 1 bootstrap requires four independent lists per method")
    rng = np.random.default_rng(seed)

    def observed(task: str) -> float:
        left = _mean([_mean(metric_vector(cell["tasks"][task], metric)) for cell in left_cells])
        right = _mean([_mean(metric_vector(cell["tasks"][task], metric)) for cell in right_cells])
        return left - right

    observed_values = {task: observed(task) for task in TASKS}
    observed_values["ood_macro"] = _mean([observed_values[task] for task in OOD_TASKS])
    draws = {task: [] for task in (*TASKS, "ood_macro")}
    arrays = {
        task: (
            np.asarray(
                [metric_vector(cell["tasks"][task], metric) for cell in left_cells],
                dtype=np.float64,
            ),
            np.asarray(
                [metric_vector(cell["tasks"][task], metric) for cell in right_cells],
                dtype=np.float64,
            ),
        )
        for task in TASKS
    }
    chunk_size = 128
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        left_selection = rng.integers(0, len(left_cells), size=(size, len(left_cells)))
        right_selection = rng.integers(0, len(right_cells), size=(size, len(right_cells)))
        chunk_draws = {}
        for task in TASKS:
            left_array, right_array = arrays[task]
            count = left_array.shape[1]
            sampled_items = rng.integers(0, count, size=(size, count))
            left_selected = left_array[left_selection].mean(axis=1)
            right_selected = right_array[right_selection].mean(axis=1)
            left = np.take_along_axis(left_selected, sampled_items, axis=1).mean(axis=1)
            right = np.take_along_axis(right_selected, sampled_items, axis=1).mean(axis=1)
            chunk_draws[task] = left - right
            draws[task].extend(float(value) for value in chunk_draws[task])
        macro = np.mean(
            np.stack([chunk_draws[task] for task in OOD_TASKS], axis=1), axis=1
        )
        draws["ood_macro"].extend(float(value) for value in macro)
    return {
        "bootstrap_schema_version": "phase1-independent-selection-common-item-v1",
        "metric": metric,
        "replicates": replicates,
        "seed": seed,
        "selection_resampling": "independent_within_method_not_paired_by_replicate_index",
        "item_resampling": "same_sampled_record_indices_for_both_methods_within_task",
        "tasks": {
            task: {
                "point_difference": observed_values[task],
                "ci90": [_quantile(draws[task], 0.05), _quantile(draws[task], 0.95)],
                "ci95": [_quantile(draws[task], 0.025), _quantile(draws[task], 0.975)],
            }
            for task in (*TASKS, "ood_macro")
        },
    }


def threshold_diagnostic(
    *, point: float, ci90: Sequence[float], ci95: Sequence[float], threshold: float = 0.01
) -> str:
    if point >= threshold and ci95[0] > 0:
        return "supports_practical_gain_threshold"
    if ci90[0] >= -threshold and ci90[1] <= threshold:
        return "supports_approximate_equivalence_band"
    if point <= -threshold and ci95[1] < 0:
        return "supports_stable_harm_threshold"
    return "insufficient_evidence"


def comparison_report(
    *,
    cells: Sequence[Mapping[str, Any]],
    replicates: int = 10_000,
    seed: int = 20260825,
) -> dict[str, Any]:
    output = {}
    for offset, (name, (left_method, right_method)) in enumerate(COMPARISONS.items()):
        left = [cell for cell in cells if cell["method"] == left_method]
        right = [cell for cell in cells if cell["method"] == right_method]
        accuracy = independent_selection_item_bootstrap(
            left_cells=left,
            right_cells=right,
            metric="accuracy",
            replicates=replicates,
            seed=seed + offset * 10,
        )
        strict_parse = independent_selection_item_bootstrap(
            left_cells=left,
            right_cells=right,
            metric="strict_parse_rate",
            replicates=replicates,
            seed=seed + offset * 10 + 1,
        )
        gsm8k = accuracy["tasks"]["gsm8k"]
        output[name] = {
            "left_method": left_method,
            "right_method": right_method,
            "accuracy": accuracy,
            "strict_parse_rate": strict_parse,
            "primary_gsm8k_threshold_diagnostic": threshold_diagnostic(
                point=float(gsm8k["point_difference"]),
                ci90=gsm8k["ci90"],
                ci95=gsm8k["ci95"],
            ),
            "publication_claim_permitted_from_phase1a_alone": False,
        }
    return {
        "analysis_schema_version": "budget-equivalent-phase1a-unblinded-v1",
        "primary_metric": "gsm8k_exact_numeric_accuracy",
        "practical_effect_threshold": 0.01,
        "selection_replicates_per_method": 4,
        "training_seeds_per_selection": 1,
        "phase1a_is_directional_pilot_not_confirmatory": True,
        "comparisons": output,
    }
