from __future__ import annotations

from eg_sft.eval.metrics import numeric_equal

ERROR_TYPES = (
    "correct",
    "arithmetic_error",
    "wrong_formula",
    "unit_scale_error",
    "temporal_ordering",
    "parse_failure",
    "variable_binding",
)


def classify_error(
    task_family: str,
    answer: float,
    parsed_prediction: float | None,
    raw_prediction: str,
) -> str:
    if parsed_prediction is None:
        return "parse_failure"
    if numeric_equal(parsed_prediction, answer):
        return "correct"
    if answer != 0:
        ratio = abs(parsed_prediction / answer)
        if 9.5 <= ratio <= 10.5 or 0.095 <= ratio <= 0.105 or 95 <= ratio <= 105:
            return "unit_scale_error"
    if task_family == "temporal_numeric_constraint":
        return "temporal_ordering"
    lowered = raw_prediction.lower()
    if any(term in lowered for term in ["average", "sum", "multiply", "percent"]):
        return "wrong_formula"
    if abs(parsed_prediction - answer) <= max(5.0, abs(answer) * 0.05):
        return "arithmetic_error"
    return "variable_binding"
