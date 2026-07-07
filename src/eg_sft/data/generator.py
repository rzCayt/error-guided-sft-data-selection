from __future__ import annotations

import random
from collections.abc import Iterable

from eg_sft.data.schemas import Buckets, Difficulty, Example, SplitName, TaskFamily
from eg_sft.data.solver import solve

TASK_FAMILIES: tuple[TaskFamily, ...] = (
    "ratio_change",
    "multiplicative_relation",
    "weighted_aggregation",
    "temporal_numeric_constraint",
)

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

SPLIT_SIZES: dict[SplitName, int] = {
    "candidate_pool": 500,
    "dev_diagnostic": 100,
    "test_id": 100,
    "test_ood_template": 50,
    "test_ood_range": 50,
}


def magnitude_bucket(answer: float) -> str:
    value = abs(answer)
    if value < 100:
        return "small"
    if value < 1000:
        return "medium"
    return "large"


def reasoning_length_bucket(task_family: str, difficulty: str) -> str:
    if difficulty == "hard" or task_family in {"weighted_aggregation", "temporal_numeric_constraint"}:
        return "long"
    if difficulty == "medium":
        return "medium"
    return "short"


def _range_for_split(split: SplitName, ood_range: bool) -> tuple[int, int]:
    if split == "test_ood_range" or ood_range:
        return 80, 600
    return 10, 180


def _ratio_params(rng: random.Random, split: SplitName) -> dict[str, object]:
    lo, hi = _range_for_split(split, False)
    return {
        "base": rng.randint(lo, hi),
        "pct": rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 40]),
        "direction": rng.choice(["increase", "decrease"]),
    }


def _multiplicative_params(rng: random.Random, split: SplitName, difficulty: Difficulty) -> dict[str, object]:
    max_factor = 18 if split != "test_ood_range" else 35
    factor_count = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    return {"factors": [rng.randint(2, max_factor) for _ in range(factor_count)]}


def _weighted_params(rng: random.Random, split: SplitName, difficulty: Difficulty) -> dict[str, object]:
    n = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    raw = [rng.randint(1, 9) for _ in range(n)]
    total = sum(raw)
    weights = [round(x / total, 3) for x in raw]
    weights[-1] = round(1 - sum(weights[:-1]), 3)
    hi = 120 if split != "test_ood_range" else 500
    return {"weights": weights, "values": [rng.randint(20, hi) for _ in range(n)]}


def _temporal_params(rng: random.Random, split: SplitName, difficulty: Difficulty) -> dict[str, object]:
    lo, hi = _range_for_split(split, split == "test_ood_range")
    steps = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    params: dict[str, object] = {
        "start": rng.randint(lo, hi),
        "deltas": [rng.randint(-30, 45) for _ in range(steps)],
        "floor": None,
        "cap": None,
    }
    if difficulty == "hard":
        params["floor"] = rng.choice([0, 20, 50])
        params["cap"] = rng.choice([220, 300, 650]) if split == "test_ood_range" else rng.choice([120, 180, 220])
    return params


def build_prompt(task_family: str, params: dict[str, object], split: SplitName) -> str:
    ood_template = split == "test_ood_template"
    if task_family == "ratio_change":
        noun = "inventory index" if ood_template else "metric"
        return (
            f"A {noun} starts at {params['base']} and then has a {params['pct']}% "
            f"{params['direction']}. What is the final value?"
        )
    if task_family == "multiplicative_relation":
        factors = params["factors"]
        if ood_template:
            return f"A nested bundle has layer sizes {factors}. What is the total count?"
        return "Multiply the related counts " + ", ".join(map(str, factors)) + ". What is the product?"
    if task_family == "weighted_aggregation":
        pairs = ", ".join(
            f"weight {w} with value {v}"
            for w, v in zip(params["weights"], params["values"], strict=True)
        )
        prefix = "A scoring table reports" if ood_template else "A weighted metric uses"
        return f"{prefix} {pairs}. What is the weighted aggregate?"
    if task_family == "temporal_numeric_constraint":
        prompt = f"Start from {params['start']} and apply ordered changes {params['deltas']}."
        if params.get("floor") is not None:
            prompt += f" The value cannot go below {params['floor']}."
        if params.get("cap") is not None:
            prompt += f" The value cannot exceed {params['cap']}."
        return prompt + " What is the final value?"
    raise ValueError(task_family)


def make_example(index: int, split: SplitName, rng: random.Random) -> Example:
    task_family = TASK_FAMILIES[index % len(TASK_FAMILIES)]
    difficulty = DIFFICULTIES[(index // len(TASK_FAMILIES)) % len(DIFFICULTIES)]
    if task_family == "ratio_change":
        params = _ratio_params(rng, split)
    elif task_family == "multiplicative_relation":
        params = _multiplicative_params(rng, split, difficulty)
    elif task_family == "weighted_aggregation":
        params = _weighted_params(rng, split, difficulty)
    else:
        params = _temporal_params(rng, split, difficulty)

    answer, rationale = solve(task_family, params)
    buckets = Buckets(
        difficulty_bucket=difficulty,
        answer_magnitude_bucket=magnitude_bucket(answer),
        reasoning_length_bucket=reasoning_length_bucket(task_family, difficulty),
    )
    return Example(
        id=f"{split}-{index:04d}",
        split=split,
        task_family=task_family,
        difficulty=difficulty,
        prompt=build_prompt(task_family, params, split),
        answer=answer,
        rationale=rationale,
        metadata={"params": params},
        buckets=buckets,
    )


def generate_split(split: SplitName, n: int | None = None, seed: int = 20260707) -> list[Example]:
    size = SPLIT_SIZES[split] if n is None else n
    rng = random.Random(seed + sum(ord(ch) for ch in split))
    return [make_example(i, split, rng) for i in range(size)]


def generate_all(seed: int = 20260707) -> dict[SplitName, list[Example]]:
    return {split: generate_split(split, seed=seed) for split in SPLIT_SIZES}


def iter_training_records(examples: Iterable[Example]) -> Iterable[dict[str, str]]:
    for ex in examples:
        yield {
            "instruction": ex.prompt,
            "response": f"{ex.rationale} Answer: {ex.answer}",
        }
