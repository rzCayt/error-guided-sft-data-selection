"""Frozen prompt and row-level scoring for GSM8K generation."""

from __future__ import annotations

from typing import Any

from eg_sft.gsm8k.parser import parse_generated_answer, parse_gold_answer


PROMPT_VERSION = "gsm8k_base_completion_v1"
PROMPT_TEMPLATE = """Solve the following grade-school math problem.
Show the calculation clearly. Your final non-empty line must use exactly this format:
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
    prediction = parse_generated_answer(generated_text)
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
        "parse_status": prediction.status,
        "parsed_prediction": (
            str(prediction.value) if prediction.value is not None else None
        ),
        "gold_value": str(gold.value),
        "numeric_correct": correct,
    }
