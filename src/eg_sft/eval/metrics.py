from __future__ import annotations


def numeric_equal(prediction: float | None, answer: float, tolerance: float = 1e-3) -> bool:
    if prediction is None:
        return False
    return abs(float(prediction) - float(answer)) <= tolerance


def absolute_percentage_error(prediction: float | None, answer: float) -> float | None:
    if prediction is None or answer == 0:
        return None
    return abs((float(prediction) - float(answer)) / float(answer))
