"""Utilities for the fixed 16-example GSM8K LoRA overfit check."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from eg_sft.gsm8k.parser import parse_gold_answer
from eg_sft.training.response_only import tokenize_response_only


PROMPT_TEMPLATE = """Solve the following grade-school math problem.
Show your reasoning, then end with exactly:
Final answer: <number>

Problem:
{question}

Solution:
"""


def format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def gsm8k_training_text(question: str, answer: str) -> tuple[str, str]:
    """Convert one canonical GSM8K row to the frozen prompt/response format."""

    parsed = parse_gold_answer(answer)
    if not parsed.ok or parsed.value is None:
        raise ValueError(f"cannot parse GSM8K gold answer: {parsed.status}")
    reasoning = answer.rsplit("####", maxsplit=1)[0].strip()
    if not reasoning:
        raise ValueError("GSM8K reasoning is empty")

    prompt = PROMPT_TEMPLATE.format(question=question.strip())
    response = f"{reasoning}\nFinal answer: {format_decimal(parsed.value)}"
    return prompt, response


def build_tokenized_overfit_examples(
    *,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    record_ids: Sequence[str],
    max_length: int,
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]]]:
    if len(rows) != len(record_ids):
        raise ValueError("rows and record_ids must have equal lengths")
    if not rows:
        raise ValueError("at least one row is required")

    examples: list[dict[str, list[int]]] = []
    audit_rows: list[dict[str, Any]] = []
    for row, record_id in zip(rows, record_ids, strict=True):
        prompt, response = gsm8k_training_text(row["question"], row["answer"])
        tokenized = tokenize_response_only(
            tokenizer,
            prompt=prompt,
            response=response,
            max_length=max_length,
            add_eos=True,
        )
        if len(tokenized["input_ids"]) >= max_length:
            raise ValueError(
                f"{record_id} reaches max_length={max_length}; "
                "the final answer may be truncated"
            )
        supervised_tokens = sum(label != -100 for label in tokenized["labels"])
        examples.append(tokenized)
        audit_rows.append(
            {
                "record_id": record_id,
                "total_tokens": len(tokenized["input_ids"]),
                "supervised_tokens": supervised_tokens,
            }
        )
    return examples, audit_rows
