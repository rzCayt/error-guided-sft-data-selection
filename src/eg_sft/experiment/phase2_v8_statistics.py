"""Crossed-block statistics for the clean 24-cell common-mix v8 study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from eg_sft.experiment.budget_equivalent_phase1_analysis import (
    OOD_TASKS,
    TASKS,
    metric_vector,
    validate_task_alignment,
)


METHODS = ("random_common_mix", "rds_error_common_mix")
REPLICATES = (1, 2, 3, 4)
TRAIN_SEEDS = (17, 29, 41)


def validate_clean_cells(cells: Sequence[Mapping[str, Any]]) -> None:
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
    if len(cells) != 24 or observed != expected:
        raise ValueError("v8 primary analysis requires the exact clean 24-cell crossing")
    validate_task_alignment(cells)


def effect_diagnostic(
    *,
    point: float,
    ci90: Sequence[float],
    ci95: Sequence[float],
    delta: float = 0.01,
    equivalence_allowed: bool = False,
) -> str:
    if ci95[0] > delta:
        return "meaningful_gain_at_least_delta"
    if ci95[0] > 0:
        return "directional_gain_below_proven_delta"
    if ci90[0] >= -delta and ci90[1] <= delta:
        return (
            "equivalent_within_delta"
            if equivalence_allowed
            else "exploratory_equivalence_signal_only"
        )
    if ci95[1] < -delta:
        return "meaningful_harm_at_least_delta"
    if ci95[1] < 0:
        return "directional_harm_below_proven_delta"
    return "insufficient_evidence"


def _arrays(
    *, cells: Sequence[Mapping[str, Any]], task: str, metric: str
) -> dict[str, np.ndarray]:
    output = {}
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
                    raise ValueError("v8 factor cell is absent or duplicated")
                seed_rows.append(metric_vector(matches[0]["tasks"][task], metric))
            rows.append(seed_rows)
        output[method] = np.asarray(rows, dtype=np.float64)
    return output


def _draw_mean(
    *,
    array: np.ndarray,
    list_indices: np.ndarray,
    global_seed_indices: np.ndarray,
    item_indices: np.ndarray,
) -> np.ndarray:
    selected_lists = array[list_indices]
    seed_index = np.broadcast_to(
        global_seed_indices[:, None, :, None],
        (
            global_seed_indices.shape[0],
            selected_lists.shape[1],
            global_seed_indices.shape[1],
            selected_lists.shape[3],
        ),
    )
    selected = np.take_along_axis(selected_lists, seed_index, axis=2)
    per_item = selected.mean(axis=(1, 2))
    return np.take_along_axis(per_item, item_indices, axis=1).mean(axis=1)


def _common_bootstrap(
    *,
    cells: Sequence[Mapping[str, Any]],
    metric: str,
    bootstrap_replicates: int = 20_000,
    seed: int = 20260828,
    resample_training_seeds: bool,
) -> dict[str, Any]:
    validate_clean_cells(cells)
    if bootstrap_replicates < 20_000:
        raise ValueError("v8 formal bootstrap requires at least 20,000 draws")
    rng = np.random.default_rng(seed)
    task_draws: dict[str, list[float]] = {
        task: [] for task in (*TASKS, "ood_macro")
    }
    observed = {}
    task_arrays = {}
    for task in TASKS:
        arrays = _arrays(cells=cells, task=task, metric=metric)
        task_arrays[task] = arrays
        observed[task] = float(
            arrays["rds_error_common_mix"].mean()
            - arrays["random_common_mix"].mean()
        )
    for start in range(0, bootstrap_replicates, 32):
        size = min(32, bootstrap_replicates - start)
        # These three draws represent the trained-cell block and must be shared
        # across GSM8K and every OOD task in the same bootstrap replicate.
        global_seeds = (
            rng.integers(0, 3, size=(size, 3))
            if resample_training_seeds
            else np.broadcast_to(np.arange(3, dtype=int), (size, 3)).copy()
        )
        random_lists = rng.integers(0, 4, size=(size, 4))
        rds_lists = rng.integers(0, 4, size=(size, 4))
        chunk_effects = {}
        for task in TASKS:
            arrays = task_arrays[task]
            count = arrays["random_common_mix"].shape[2]
            # Item draws remain task-specific because tasks have different IDs.
            items = rng.integers(0, count, size=(size, count))
            random_mean = _draw_mean(
                array=arrays["random_common_mix"],
                list_indices=random_lists,
                global_seed_indices=global_seeds,
                item_indices=items,
            )
            rds_mean = _draw_mean(
                array=arrays["rds_error_common_mix"],
                list_indices=rds_lists,
                global_seed_indices=global_seeds,
                item_indices=items,
            )
            effects = rds_mean - random_mean
            chunk_effects[task] = effects
            task_draws[task].extend(effects.tolist())
        ood_chunk = np.mean(
            np.stack([chunk_effects[task] for task in OOD_TASKS], axis=0),
            axis=0,
        )
        task_draws["ood_macro"].extend(ood_chunk.tolist())
    observed["ood_macro"] = float(np.mean([observed[task] for task in OOD_TASKS]))
    results = {}
    for task in (*TASKS, "ood_macro"):
        values = np.asarray(task_draws[task], dtype=np.float64)
        ci90 = [float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))]
        ci95 = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        results[task] = {
            "point_difference": observed[task],
            "ci90": ci90,
            "ci95": ci95,
            "effect_diagnostic": effect_diagnostic(
                point=observed[task], ci90=ci90, ci95=ci95
            ),
        }
    return {
        "schema_version": (
            "phase2-v8-seed-resampled-bootstrap-v1"
            if resample_training_seeds
            else "phase2-v8-fixed-seed-block-bootstrap-v1"
        ),
        "metric": metric,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "training_seed_resampling": (
            "one_global_draw_shared_across_all_cells_and_tasks"
            if resample_training_seeds
            else "fixed_observed_seed_blocks_shared_across_all_cells_and_tasks"
        ),
        "random_list_resampling": "independent_within_random",
        "rds_list_resampling": "independent_within_rds",
        "random_and_rds_list_indices_paired": False,
        "item_resampling": (
            "one_task_specific_draw_shared_across_all_cells_within_task"
        ),
        "ood_aggregation": (
            "dataset_equal_macro_with_shared_list_and_seed_draws"
        ),
        "delta": 0.01,
        "results": results,
    }


def fixed_seed_common_bootstrap(
    *,
    cells: Sequence[Mapping[str, Any]],
    metric: str,
    bootstrap_replicates: int = 20_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    """Primary conditional analysis: retain all three observed seed blocks."""

    return _common_bootstrap(
        cells=cells,
        metric=metric,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        resample_training_seeds=False,
    )


def crossed_common_bootstrap(
    *,
    cells: Sequence[Mapping[str, Any]],
    metric: str,
    bootstrap_replicates: int = 20_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    """Exploratory sensitivity treating the three seeds as sampled blocks."""

    return _common_bootstrap(
        cells=cells,
        metric=metric,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        resample_training_seeds=True,
    )


def seed_block_effects(
    *, cells: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    """Report method effects separately for every observed training seed."""

    validate_clean_cells(cells)
    rows = []
    for seed in TRAIN_SEEDS:
        by_task = {}
        for task in TASKS:
            rds = [
                np.mean(metric_vector(row["tasks"][task], metric))
                for row in cells
                if row["method"] == "rds_error_common_mix"
                and int(row["train_seed"]) == seed
            ]
            random = [
                np.mean(metric_vector(row["tasks"][task], metric))
                for row in cells
                if row["method"] == "random_common_mix"
                and int(row["train_seed"]) == seed
            ]
            by_task[task] = float(np.mean(rds) - np.mean(random))
        by_task["ood_macro"] = float(np.mean([by_task[task] for task in OOD_TASKS]))
        rows.append({"train_seed": seed, "effects": by_task})
    return {
        "schema_version": "phase2-v8-seed-block-effects-v1",
        "metric": metric,
        "training_seed_role": "fixed_observed_blocks",
        "rows": rows,
    }


def complete_leave_one_out(
    *, cells: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    validate_clean_cells(cells)
    outputs = {"leave_one_seed_out": [], "leave_one_random_list_out": [], "leave_one_rds_list_out": []}

    def task_effect(subset: Sequence[Mapping[str, Any]], task: str) -> float:
        left = [row for row in subset if row["method"] == "rds_error_common_mix"]
        right = [row for row in subset if row["method"] == "random_common_mix"]
        return float(
            np.mean([np.mean(metric_vector(row["tasks"][task], metric)) for row in left])
            - np.mean([np.mean(metric_vector(row["tasks"][task], metric)) for row in right])
        )

    for seed in TRAIN_SEEDS:
        subset = [row for row in cells if int(row["train_seed"]) != seed]
        outputs["leave_one_seed_out"].append(
            {"omitted_seed": seed, "gsm8k_difference": task_effect(subset, "gsm8k")}
        )
    for replicate in REPLICATES:
        random_subset = [
            row
            for row in cells
            if not (
                row["method"] == "random_common_mix"
                and int(row["replicate_index"]) == replicate
            )
        ]
        rds_subset = [
            row
            for row in cells
            if not (
                row["method"] == "rds_error_common_mix"
                and int(row["replicate_index"]) == replicate
            )
        ]
        outputs["leave_one_random_list_out"].append(
            {"omitted_list": replicate, "gsm8k_difference": task_effect(random_subset, "gsm8k")}
        )
        outputs["leave_one_rds_list_out"].append(
            {"omitted_list": replicate, "gsm8k_difference": task_effect(rds_subset, "gsm8k")}
        )
    return {
        "schema_version": "phase2-v8-complete-leave-one-out-v1",
        "metric": metric,
        **outputs,
    }


def precision_sensitivity_grid(
    *,
    list_sd_values: Sequence[float] = (0.005, 0.01, 0.02),
    selector_seed_interaction_sd_values: Sequence[float] = (0.01, 0.0219, 0.03),
    rds_effective_list_counts: Sequence[float] = (1.5, 2.0, 3.0, 4.0),
    effects: Sequence[float] = (-0.01, 0.0, 0.01, 0.02),
    simulations: int = 20_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    if simulations < 20_000:
        raise ValueError("precision simulation requires at least 20,000 draws")
    rng = np.random.default_rng(seed)
    rows = []
    z95 = 1.959963984540054
    z90 = 1.6448536269514722
    for list_sd in list_sd_values:
        for seed_sd in selector_seed_interaction_sd_values:
            for effective_rds in rds_effective_list_counts:
                standard_error = float(
                    np.sqrt(
                        list_sd**2 / 4
                        + list_sd**2 / effective_rds
                        + 2 * seed_sd**2 / 3
                    )
                )
                for effect in effects:
                    estimates = rng.normal(effect, standard_error, size=simulations)
                    lower95 = estimates - z95 * standard_error
                    upper95 = estimates + z95 * standard_error
                    lower90 = estimates - z90 * standard_error
                    upper90 = estimates + z90 * standard_error
                    rows.append(
                        {
                            "list_sd": float(list_sd),
                            "selector_seed_interaction_sd": float(seed_sd),
                            "rds_effective_list_count": float(effective_rds),
                            "true_effect": float(effect),
                            "standard_error": standard_error,
                            "median_ci95_half_width": z95 * standard_error,
                            "directional_gain_probability": float(np.mean(lower95 > 0)),
                            "meaningful_gain_probability": float(np.mean(lower95 > 0.01)),
                            "equivalence_probability": float(
                                np.mean((lower90 >= -0.01) & (upper90 <= 0.01))
                            ),
                            "meaningful_harm_probability": float(np.mean(upper95 < -0.01)),
                        }
                    )
    worst_relevant = [
        row
        for row in rows
        if row["selector_seed_interaction_sd"] == 0.0219
        and row["rds_effective_list_count"] <= 2.0
    ]
    equivalence_feasible = max(row["equivalence_probability"] for row in worst_relevant) >= 0.8
    return {
        "schema_version": "phase2-v8-prospective-precision-simulation-v1",
        "status": "PASS",
        "simulations_per_scenario": simulations,
        "seed": seed,
        "assumption": "normal random-effects sensitivity grid; not a power guarantee",
        "pilot_selector_seed_interaction_sd_reference": 0.0219,
        "equivalence_feasible_under_high_overlap_reference": equivalence_feasible,
        "equivalence_status": "PRIMARY_CAPABLE" if equivalence_feasible else "EXPLORATORY_ONLY",
        "rows": rows,
    }
