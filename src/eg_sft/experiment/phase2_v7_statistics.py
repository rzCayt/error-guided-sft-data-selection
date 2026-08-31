"""Pre-registered 4-list x 3-seed hierarchical statistics for Phase-2 v7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from eg_sft.experiment.budget_equivalent_phase1_analysis import (
    OOD_TASKS,
    TASKS,
    metric_vector,
    threshold_diagnostic,
    validate_task_alignment,
)


METHODS = (
    "random_common_mix",
    "rds_error_common_mix",
    "random_free_mix",
    "rds_error_free_mix",
)
REPLICATES = (1, 2, 3, 4)
TRAIN_SEEDS = (17, 29, 41)


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def validate_confirmatory_cells(cells: Sequence[Mapping[str, Any]]) -> None:
    observed = {
        (str(row["method"]), int(row["replicate_index"]), int(row["train_seed"]))
        for row in cells
    }
    expected = {
        (method, replicate, seed)
        for method in METHODS
        for replicate in REPLICATES
        for seed in TRAIN_SEEDS
    }
    if len(cells) != 48 or observed != expected:
        raise ValueError("confirmatory analysis requires the exact 48-cell crossing")
    validate_task_alignment(cells)


def _method_arrays(
    *, cells: Sequence[Mapping[str, Any]], task: str, metric: str
) -> dict[str, np.ndarray]:
    arrays = {}
    for method in METHODS:
        rows = []
        for replicate in REPLICATES:
            seed_rows = []
            for seed in TRAIN_SEEDS:
                matches = [
                    row
                    for row in cells
                    if row["method"] == method
                    and int(row["replicate_index"]) == replicate
                    and int(row["train_seed"]) == seed
                ]
                if len(matches) != 1:
                    raise ValueError("factor cell is absent or duplicated")
                seed_rows.append(metric_vector(matches[0]["tasks"][task], metric))
            rows.append(seed_rows)
        arrays[method] = np.asarray(rows, dtype=np.float64)
    return arrays


def _draw_method_item_means(
    *,
    array: np.ndarray,
    list_indices: np.ndarray,
    seed_indices: np.ndarray,
    item_indices: np.ndarray,
) -> np.ndarray:
    # array: list x seed x item. Lists and seeds are resampled within method;
    # item indices are shared across methods for the same task/draw.
    selected_lists = array[list_indices]
    selected = np.take_along_axis(
        selected_lists,
        seed_indices[..., None],
        axis=2,
    )
    per_item = selected.mean(axis=(1, 2))
    return np.take_along_axis(per_item, item_indices, axis=1).mean(axis=1)


def hierarchical_four_method_bootstrap(
    *,
    cells: Sequence[Mapping[str, Any]],
    metric: str,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    validate_confirmatory_cells(cells)
    if bootstrap_replicates < 10_000:
        raise ValueError("formal Phase-2 bootstrap requires at least 10,000 draws")
    rng = np.random.default_rng(seed)
    estimates = {
        task: {
            "common_rds_minus_random": [],
            "free_rds_minus_random": [],
            "free_minus_common_interaction": [],
        }
        for task in (*TASKS, "ood_macro")
    }
    observed = {}
    for task in TASKS:
        arrays = _method_arrays(cells=cells, task=task, metric=metric)
        means = {method: float(array.mean()) for method, array in arrays.items()}
        observed[task] = {
            "common_rds_minus_random": (
                means["rds_error_common_mix"] - means["random_common_mix"]
            ),
            "free_rds_minus_random": (
                means["rds_error_free_mix"] - means["random_free_mix"]
            ),
        }
        observed[task]["free_minus_common_interaction"] = (
            observed[task]["free_rds_minus_random"]
            - observed[task]["common_rds_minus_random"]
        )
        count = next(iter(arrays.values())).shape[2]
        chunk = 32
        for start in range(0, bootstrap_replicates, chunk):
            size = min(chunk, bootstrap_replicates - start)
            item_indices = rng.integers(0, count, size=(size, count))
            draws = {}
            for method, array in arrays.items():
                list_indices = rng.integers(0, 4, size=(size, 4))
                seed_indices = rng.integers(0, 3, size=(size, 4, 3))
                draws[method] = _draw_method_item_means(
                    array=array,
                    list_indices=list_indices,
                    seed_indices=seed_indices,
                    item_indices=item_indices,
                )
            common = draws["rds_error_common_mix"] - draws["random_common_mix"]
            free = draws["rds_error_free_mix"] - draws["random_free_mix"]
            interaction = free - common
            estimates[task]["common_rds_minus_random"].extend(common.tolist())
            estimates[task]["free_rds_minus_random"].extend(free.tolist())
            estimates[task]["free_minus_common_interaction"].extend(
                interaction.tolist()
            )
    observed["ood_macro"] = {
        estimand: float(np.mean([observed[task][estimand] for task in OOD_TASKS]))
        for estimand in (
            "common_rds_minus_random",
            "free_rds_minus_random",
            "free_minus_common_interaction",
        )
    }
    for index in range(bootstrap_replicates):
        for estimand in observed["ood_macro"]:
            estimates["ood_macro"][estimand].append(
                float(np.mean([estimates[task][estimand][index] for task in OOD_TASKS]))
            )
    results = {}
    for task in (*TASKS, "ood_macro"):
        results[task] = {}
        for estimand, values in estimates[task].items():
            array = np.asarray(values, dtype=np.float64)
            point = float(observed[task][estimand])
            ci90 = [_quantile(array, 0.05), _quantile(array, 0.95)]
            ci95 = [_quantile(array, 0.025), _quantile(array, 0.975)]
            results[task][estimand] = {
                "point_difference": point,
                "ci90": ci90,
                "ci95": ci95,
                "threshold_diagnostic": threshold_diagnostic(
                    point=point, ci90=ci90, ci95=ci95, threshold=0.01
                ),
            }
    return {
        "schema_version": "phase2-v7-hierarchical-bootstrap-v1",
        "metric": metric,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "selection_resampling": "independent_within_each_method",
        "training_seed_resampling": "within_sampled_selection_list",
        "item_resampling": "shared_across_all_methods_within_task",
        "ood_aggregation": "equal_weight_macro_not_pooled_items",
        "minimum_meaningful_effect": 0.01,
        "results": results,
    }


def descriptive_variance_components(
    *, cells: Sequence[Mapping[str, Any]], task: str, metric: str
) -> dict[str, Any]:
    validate_confirmatory_cells(cells)
    arrays = _method_arrays(cells=cells, task=task, metric=metric)
    output = {}
    for method, item_array in arrays.items():
        values = item_array.mean(axis=2)
        grand = float(values.mean())
        list_means = values.mean(axis=1)
        seed_means = values.mean(axis=0)
        interaction = values - list_means[:, None] - seed_means[None, :] + grand
        output[method] = {
            "between_selection_list_variance_of_seed_means": float(
                np.var(list_means, ddof=1)
            ),
            "between_training_seed_variance_of_list_means": float(
                np.var(seed_means, ddof=1)
            ),
            "list_by_seed_residual_mean_square": float(np.mean(interaction**2)),
            "training_seed_treated_as_fixed_block_in_model_sensitivity": True,
        }
    return {
        "schema_version": "phase2-v7-descriptive-variance-v1",
        "task": task,
        "metric": metric,
        "components_are_descriptive_not_causal": True,
        "methods": output,
    }
