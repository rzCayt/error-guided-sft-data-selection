from decimal import Decimal

import pytest

from eg_sft.evaluation.arithmetic_ood import (
    build_ood_prompt,
    build_ood_record,
    parse_unique_numeric_gold,
    score_ood_generation,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("9 (apples)", Decimal("9")),
        ("-3.50 dollars", Decimal("-3.50")),
        ("1,234", Decimal("1234")),
    ],
)
def test_unique_numeric_gold_accepts_one_number(raw: str, expected: Decimal) -> None:
    value, status = parse_unique_numeric_gold(raw)
    assert status == "ok"
    assert value == expected


@pytest.mark.parametrize("raw", ["Yes", "Mrs. Hilt", "3:30 p.m.", "2 or 3"])
def test_unique_numeric_gold_rejects_non_numeric_or_ambiguous(raw: str) -> None:
    value, status = parse_unique_numeric_gold(raw)
    assert value is None
    assert status != "ok"


def test_source_adapters_reuse_frozen_gsm8k_prompt() -> None:
    prompt = build_ood_prompt(
        "svamp",
        {"Body": "Mia has two apples.", "Question": "She buys three. How many?"},
    )
    assert "Mia has two apples." in prompt
    assert "Final answer: <number>" in prompt


def test_ood_record_and_scoring_preserve_wrong_answer() -> None:
    record = build_ood_record(
        dataset="asdiv_numeric",
        source_index=4,
        row={"body": "There are nine apples.", "question": "How many?", "answer": "9 (apples)"},
        answer_field="answer",
    )
    assert record["numeric_eligible"] is True
    result = score_ood_generation(
        record=record,
        gold_value=record["gold_value"],
        generated_text="Mistake.\nFinal answer: 8",
    )
    assert result["parsed_prediction"] == "8"
    assert result["numeric_correct"] is False
