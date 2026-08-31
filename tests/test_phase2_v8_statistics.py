from __future__ import annotations

from eg_sft.experiment.phase2_v8_statistics import (
    METHODS,
    REPLICATES,
    TRAIN_SEEDS,
    crossed_common_bootstrap,
    fixed_seed_common_bootstrap,
    seed_block_effects,
    effect_diagnostic,
    precision_sensitivity_grid,
    validate_clean_cells,
)


TASKS = ("gsm8k", "svamp", "asdiv_numeric", "multiarith")


def _rows(value: bool) -> list[dict]:
    return [
        {
            "record_id": f"r{index}",
            "numeric_correct": value,
            "strict_parse_status": "ok",
            "parse_status": "ok",
        }
        for index in range(5)
    ]


def _cells() -> list[dict]:
    return [
        {
            "cell_id": f"{method}-{replicate}-{seed}",
            "method": method,
            "replicate_index": replicate,
            "train_seed": seed,
            "tasks": {
                task: _rows(method == "rds_error_common_mix") for task in TASKS
            },
        }
        for method in METHODS
        for replicate in REPLICATES
        for seed in TRAIN_SEEDS
    ]


def test_v8_crossing_and_global_seed_bootstrap() -> None:
    cells = _cells()
    validate_clean_cells(cells)
    report = crossed_common_bootstrap(
        cells=cells, metric="accuracy", bootstrap_replicates=20_000, seed=7
    )
    primary = report["results"]["gsm8k"]
    assert primary["point_difference"] == 1.0
    assert primary["ci95"] == [1.0, 1.0]
    assert report["training_seed_resampling"].startswith("one_global_draw")
    assert report["ood_aggregation"].endswith("shared_list_and_seed_draws")


def test_effect_rules_do_not_call_point_above_1pp_meaningful() -> None:
    assert effect_diagnostic(
        point=0.02, ci90=[-0.001, 0.03], ci95=[0.001, 0.04]
    ) == "directional_gain_below_proven_delta"
    assert effect_diagnostic(
        point=0.03, ci90=[0.015, 0.04], ci95=[0.011, 0.05]
    ) == "meaningful_gain_at_least_delta"
    assert effect_diagnostic(
        point=0.0, ci90=[-0.009, 0.009], ci95=[-0.02, 0.02]
    ) == "exploratory_equivalence_signal_only"
    assert effect_diagnostic(
        point=0.0,
        ci90=[-0.009, 0.009],
        ci95=[-0.02, 0.02],
        equivalence_allowed=True,
    ) == "equivalent_within_delta"


def test_precision_grid_downgrades_equivalence_when_seed_noise_is_large() -> None:
    report = precision_sensitivity_grid(simulations=20_000, seed=3)
    assert report["status"] == "PASS"
    assert report["equivalence_status"] == "EXPLORATORY_ONLY"
    assert len(report["rows"]) == 3 * 3 * 4 * 4


def test_fixed_seed_primary_does_not_resample_seed_blocks() -> None:
    cells = _cells()
    report = fixed_seed_common_bootstrap(
        cells=cells, metric="accuracy", bootstrap_replicates=20_000, seed=11
    )
    assert report["training_seed_resampling"].startswith("fixed_observed")
    assert report["results"]["gsm8k"]["point_difference"] == 1.0
    seed_rows = seed_block_effects(cells=cells, metric="accuracy")["rows"]
    assert [row["train_seed"] for row in seed_rows] == [17, 29, 41]
    assert all(row["effects"]["gsm8k"] == 1.0 for row in seed_rows)
