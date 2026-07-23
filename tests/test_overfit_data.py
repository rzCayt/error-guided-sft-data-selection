from decimal import Decimal

import pytest

from eg_sft.training.overfit import format_decimal, gsm8k_training_text


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (Decimal("42"), "42"),
        (Decimal("-3.50"), "-3.5"),
        (Decimal("0.1250"), "0.125"),
    ],
)
def test_format_decimal(value: Decimal, rendered: str) -> None:
    assert format_decimal(value) == rendered


def test_gsm8k_training_text_rewrites_gold_marker() -> None:
    prompt, response = gsm8k_training_text(
        "What is 2 + 3?",
        "Add the values: 2 + 3 = 5.\n#### 5",
    )
    assert "What is 2 + 3?" in prompt
    assert "Final answer: <number>" in prompt
    assert response.endswith("Final answer: 5")
    assert "####" not in response
