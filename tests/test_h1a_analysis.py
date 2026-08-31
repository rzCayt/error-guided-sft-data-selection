import pytest

from eg_sft.experiment.h1a_analysis import (
    average_ranks,
    fixed_count_permutations,
    partial_spearman,
    top_bottom_mean_difference,
)


def test_average_ranks_handles_ties() -> None:
    assert average_ranks([10, 20, 20, 40]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_partial_spearman_recovers_increment_after_control() -> None:
    control = list(range(1, 21))
    predictor = [value + (index % 3) for index, value in enumerate(control)]
    outcome = [
        2 * control[index] + 5 * predictor[index]
        for index in range(len(control))
    ]
    rho = partial_spearman(
        predictor=predictor,
        outcome=outcome,
        controls=[control],
    )
    assert rho > 0.8


def test_fixed_count_permutations_are_deterministic_and_preserve_count() -> None:
    first = fixed_count_permutations(
        item_count=20,
        selected_count=5,
        permutation_count=10,
        seed=20260722,
    )
    second = fixed_count_permutations(
        item_count=20,
        selected_count=5,
        permutation_count=10,
        seed=20260722,
    )
    assert first == second
    assert all(len(indices) == len(set(indices)) == 5 for indices in first)


def test_partial_spearman_returns_zero_without_incremental_rank_variation() -> None:
    control = list(range(10))
    assert partial_spearman(
        predictor=control,
        outcome=list(reversed(control)),
        controls=[control],
    ) == pytest.approx(0.0)


def test_top_bottom_difference_uses_score_extremes() -> None:
    difference, top, bottom = top_bottom_mean_difference(
        scores=[1.0, 0.8, 0.2, 0.0],
        utilities=[4.0, 2.0, -1.0, -3.0],
        group_count=2,
    )
    assert top == [0, 1]
    assert bottom == [2, 3]
    assert difference == pytest.approx(5.0)
