from __future__ import annotations

from eg_sft.experiment.phase2_v7_statistics import (
    METHODS,
    REPLICATES,
    TRAIN_SEEDS,
    descriptive_variance_components,
    hierarchical_four_method_bootstrap,
    validate_confirmatory_cells,
)


TASKS = ("gsm8k", "svamp", "asdiv_numeric", "multiarith")


def _rows(value: bool) -> list[dict]:
    return [
        {
            "record_id": f"r{index}",
            "numeric_correct": value,
            "strict_parse_status": "ok" if value else "missing_final_marker",
            "parse_status": "ok",
        }
        for index in range(5)
    ]


def _cells() -> list[dict]:
    output = []
    for method in METHODS:
        value = method.startswith("rds_error")
        for replicate in REPLICATES:
            for seed in TRAIN_SEEDS:
                output.append(
                    {
                        "cell_id": f"{method}-{replicate}-{seed}",
                        "method": method,
                        "replicate_index": replicate,
                        "train_seed": seed,
                        "tasks": {task: _rows(value) for task in TASKS},
                    }
                )
    return output


def test_confirmatory_crossing_requires_all_48_cells() -> None:
    cells = _cells()
    validate_confirmatory_cells(cells)
    try:
        validate_confirmatory_cells(cells[:-1])
    except ValueError as error:
        assert "48-cell" in str(error)
    else:
        raise AssertionError("missing confirmatory cell must fail")


def test_hierarchical_bootstrap_uses_lists_seeds_and_equal_ood_macro() -> None:
    report = hierarchical_four_method_bootstrap(
        cells=_cells(), metric="accuracy", bootstrap_replicates=10_000, seed=7
    )
    common = report["results"]["gsm8k"]["common_rds_minus_random"]
    assert common["point_difference"] == 1.0
    assert common["ci95"] == [1.0, 1.0]
    assert report["results"]["ood_macro"]["common_rds_minus_random"] == common
    assert report["selection_resampling"] == "independent_within_each_method"


def test_variance_decomposition_reports_three_distinct_sources() -> None:
    report = descriptive_variance_components(
        cells=_cells(), task="gsm8k", metric="accuracy"
    )
    row = report["methods"]["random_common_mix"]
    assert "between_selection_list_variance_of_seed_means" in row
    assert "between_training_seed_variance_of_list_means" in row
    assert "list_by_seed_residual_mean_square" in row
