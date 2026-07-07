from __future__ import annotations

from typing import Any


def round_answer(value: float) -> float:
    return round(float(value), 4)


def solve_ratio_change(params: dict[str, Any]) -> tuple[float, str]:
    base = float(params["base"])
    pct = float(params["pct"])
    direction = params["direction"]
    multiplier = 1 + pct / 100 if direction == "increase" else 1 - pct / 100
    answer = round_answer(base * multiplier)
    rationale = f"Apply {direction} of {pct}%: {base} * {round(multiplier, 6)} = {answer}."
    return answer, rationale


def solve_multiplicative_relation(params: dict[str, Any]) -> tuple[float, str]:
    factors = [float(x) for x in params["factors"]]
    result = 1.0
    for factor in factors:
        result *= factor
    answer = round_answer(result)
    rationale = "Multiply the chain: " + " * ".join(str(int(x)) for x in factors) + f" = {answer}."
    return answer, rationale


def solve_weighted_aggregation(params: dict[str, Any]) -> tuple[float, str]:
    weights = [float(x) for x in params["weights"]]
    values = [float(x) for x in params["values"]]
    total = sum(w * v for w, v in zip(weights, values, strict=True))
    answer = round_answer(total)
    pieces = [f"{w}*{v}" for w, v in zip(weights, values, strict=True)]
    rationale = "Compute weighted sum: " + " + ".join(pieces) + f" = {answer}."
    return answer, rationale


def solve_temporal_numeric_constraint(params: dict[str, Any]) -> tuple[float, str]:
    value = float(params["start"])
    steps = []
    for delta in params["deltas"]:
        value += float(delta)
        if params.get("floor") is not None:
            value = max(value, float(params["floor"]))
        if params.get("cap") is not None:
            value = min(value, float(params["cap"]))
        steps.append(value)
    answer = round_answer(value)
    rationale = f"Apply ordered changes {params['deltas']} from {params['start']}; final value is {answer}."
    return answer, rationale


SOLVERS = {
    "ratio_change": solve_ratio_change,
    "multiplicative_relation": solve_multiplicative_relation,
    "weighted_aggregation": solve_weighted_aggregation,
    "temporal_numeric_constraint": solve_temporal_numeric_constraint,
}


def solve(task_family: str, params: dict[str, Any]) -> tuple[float, str]:
    if task_family not in SOLVERS:
        raise ValueError(f"Unknown task family: {task_family}")
    return SOLVERS[task_family](params)
