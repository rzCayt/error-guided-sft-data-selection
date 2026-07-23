"""Frozen prompt and row-level scoring for GSM8K generation."""

from __future__ import annotations

from typing import Any

from eg_sft.gsm8k.parser import (
    parse_generated_answer,
    parse_gold_answer,
    parse_last_numeric_answer,
)


PROMPT_VERSION = "gsm8k_base_completion_v2_one_shot_frozen"
PROMPT_TEMPLATE = """Solve the grade-school math problem by following the example format.
The final non-empty line must contain only ``Final answer:`` followed by a number.
Do not put units, currency symbols, words, or punctuation after the number.

Example problem:
Mia has 2 apples and buys 3 more. How many apples does she have?

Example solution:
Mia has 2 + 3 = 5 apples.
Final answer: 5

Now solve the next problem. Show the calculation clearly and finish with exactly:
Final answer: <number>

Problem:
{question}

Solution:
"""


def build_evaluation_prompt(question: str) -> str:
    if not question.strip():
        raise ValueError("question must be non-empty")
    return PROMPT_TEMPLATE.format(question=question.strip())


def score_generation(
    *,
    record: dict[str, Any],
    gold_answer_text: str,
    generated_text: str,
) -> dict[str, Any]:
    """Create one auditable output row without repairing model text."""

    gold = parse_gold_answer(gold_answer_text)
    if not gold.ok or gold.value is None:
        raise ValueError(f"invalid GSM8K gold answer: {gold.status}")
    strict_prediction = parse_generated_answer(generated_text)
    if strict_prediction.ok:
        prediction = strict_prediction
        parse_mode = "strict_final_marker"
    else:
        prediction = parse_last_numeric_answer(generated_text)
        parse_mode = "last_numeric_fallback" if prediction.ok else "failed"
    correct = bool(
        prediction.ok
        and prediction.value is not None
        and prediction.value == gold.value
    )
    return {
        "record_id": record["record_id"],
        "source_index": int(record["source_index"]),
        "question_sha256": record["question_sha256"],
        "prompt_version": PROMPT_VERSION,
        "raw_output": generated_text,
        "strict_parse_status": strict_prediction.status,
        "strict_parsed_prediction": (
            str(strict_prediction.value)
            if strict_prediction.value is not None
            else None
        ),
        "parse_mode": parse_mode,
        "parse_status": prediction.status,
        "parsed_prediction": (
            str(prediction.value) if prediction.value is not None else None
        ),
        "gold_value": str(gold.value),
        "numeric_correct": correct,
    }
