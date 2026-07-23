import pytest

from eg_sft.experiment.utility import (
    icc_absolute_agreement,
    pearson_correlation,
)


def test_icc_is_one_for_identical_nonconstant_repeats() -> None:
    values = [[float(index), float(index)] for index in range(10)]
    assert icc_absolute_agreement(values) == pytest.approx(1.0)


def test_icc_penalizes_repeat_disagreement() -> None:
    values = [
        [0.0, 9.0],
        [1.0, 8.0],
        [2.0, 7.0],
        [3.0, 6.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [6.0, 3.0],
        [7.0, 2.0],
        [8.0, 1.0],
        [9.0, 0.0],
    ]
    assert icc_absolute_agreement(values) < 0.0


def test_pearson_correlation_handles_scale_but_not_reversal() -> None:
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson_correlation([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
