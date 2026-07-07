from eg_sft.data.generator import generate_split
from eg_sft.selection.error_guided import select_error_guided
from eg_sft.selection.matched_random import select_matched_random, matching_key


def test_generator_is_deterministic() -> None:
    left = [row.to_dict() for row in generate_split("candidate_pool", n=8, seed=123)]
    right = [row.to_dict() for row in generate_split("candidate_pool", n=8, seed=123)]
    assert left == right
    assert {row["task_family"] for row in left}


def test_selection_budget_and_matching() -> None:
    candidates = [row.to_dict() for row in generate_split("candidate_pool", n=80, seed=123)]
    profile = [
        {
            "task_family": "temporal_numeric_constraint",
            "difficulty_bucket": "hard",
            "answer_magnitude_bucket": "medium",
            "reasoning_length_bucket": "long",
            "count": "10",
            "failures": "8",
            "error_rate": "0.8",
        }
    ]
    targeted = select_error_guided(candidates, profile, budget=16, seed=1)
    matched = select_matched_random(candidates, targeted, seed=2)
    assert len(targeted) == 16
    assert len(matched) == 16
    assert {row["id"] for row in targeted}.isdisjoint({row["id"] for row in matched})
    assert len({matching_key(row) for row in targeted}) >= 1
