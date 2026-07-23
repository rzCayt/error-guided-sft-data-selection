from decimal import Decimal

import pytest

from eg_sft.gsm8k.parser import (
    parse_generated_answer,
    parse_gold_answer,
    parse_last_numeric_answer,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reasoning\n#### 42", Decimal("42")),
        ("reasoning\n#### -3.5", Decimal("-3.5")),
        ("reasoning\n#### 1,234", Decimal("1234")),
    ],
)
def test_parse_gold_answer(text: str, expected: Decimal) -> None:
    parsed = parse_gold_answer(text)
    assert parsed.ok
    assert parsed.value == expected


def test_gold_marker_is_required() -> None:
    parsed = parse_gold_answer("The answer is 42")
    assert parsed.value is None
    assert parsed.status == "missing_gold_marker"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\nFinal answer: 42", Decimal("42")),
        ("Final answer: -3.50", Decimal("-3.50")),
        ("Final ANSWER: 1,234", Decimal("1234")),
    ],
)
def test_parse_generated_answer(text: str, expected: Decimal) -> None:
    parsed = parse_generated_answer(text)
    assert parsed.ok
    assert parsed.value == expected


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("", "empty_output"),
        ("answer is 42", "missing_final_marker"),
        ("Final answer: <number>", "missing_final_marker"),
        ("Final answer: 3 + 4", "missing_final_marker"),
        ("Final answer: 7 apples", "missing_final_marker"),
        ("Final answer: 7\nextra text", "marker_not_final"),
        ("Final answer: 1\nFinal answer: 2", "multiple_final_markers"),
    ],
)
def test_malformed_generated_answers_are_rejected(text: str, status: str) -> None:
    parsed = parse_generated_answer(text)
    assert parsed.value is None
    assert parsed.status == status


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The property is worth $333,200.", Decimal("333200")),
        ("x = 70 cm\nFinal answer: 70 cm", Decimal("70")),
        ("First 2, then 5.", Decimal("5")),
    ],
)
def test_last_numeric_fallback_is_explicit(text: str, expected: Decimal) -> None:
    parsed = parse_last_numeric_answer(text)
    assert parsed.ok
    assert parsed.value == expected


def test_last_numeric_fallback_rejects_output_without_numbers() -> None:
    parsed = parse_last_numeric_answer("No numeric answer was produced.")
    assert parsed.value is None
    assert parsed.status == "no_numeric_token"
