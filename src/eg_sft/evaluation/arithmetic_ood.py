"""Frozen adapters and numeric scoring for arithmetic OOD datasets."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from eg_sft.data.public_gsm8k import sha256_text
from eg_sft.evaluation.gsm8k_generation import PROMPT_VERSION, build_evaluation_prompt
from eg_sft.gsm8k.parser import parse_generated_answer, parse_last_numeric_answer


_NUMBER = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?")


def parse_unique_numeric_gold(value: Any) -> tuple[Decimal | None, str]:
    """Accept one integer/decimal token and reject ambiguous answer strings."""

    text = str(value).strip()
    matches = list(_NUMBER.finditer(text))
    if not matches:
        return None, "no_numeric_gold"
    if len(matches) != 1:
        return None, "ambiguous_multiple_numeric_gold"
    try:
        return Decimal(matches[0].group(0).replace(",", "")), "ok"
    except InvalidOperation:
        return None, "invalid_numeric_gold"


def source_question(dataset: str, row: dict[str, Any]) -> str:
    """Map one frozen source schema to a non-empty evaluation question."""

    if dataset == "svamp":
        parts = [str(row.get("Body", "")).strip(), str(row.get("Question", "")).strip()]
    elif dataset == "asdiv_numeric":
        parts = [str(row.get("body", "")).strip(), str(row.get("question", "")).strip()]
    elif dataset == "multiarith":
        parts = [str(row.get("question", "")).strip()]
    else:
        raise ValueError(f"unsupported arithmetic OOD dataset: {dataset}")
    question = " ".join(part for part in parts if part)
    if not question:
        raise ValueError("OOD question is empty")
    return question


def build_ood_record(
    *, dataset: str, source_index: int, row: dict[str, Any], answer_field: str
) -> dict[str, Any]:
    question = source_question(dataset, row)
    gold, status = parse_unique_numeric_gold(row.get(answer_field, ""))
    question_sha256 = sha256_text(question)
    record = {
        "record_id": f"{dataset}-{source_index:05d}-{question_sha256[:12]}",
        "dataset": dataset,
        "source_index": source_index,
        "question_sha256": question_sha256,
        "answer_sha256": sha256_text(str(row.get(answer_field, ""))),
        "gold_parse_status": status,
        "numeric_eligible": gold is not None,
    }
    if gold is not None:
        record["gold_value"] = str(gold)
    return record


def score_ood_generation(
    *, record: dict[str, Any], gold_value: str, generated_text: str
) -> dict[str, Any]:
    gold = Decimal(gold_value)
    strict = parse_generated_answer(generated_text)
    if strict.ok:
        prediction = strict
        parse_mode = "strict_final_marker"
    else:
        prediction = parse_last_numeric_answer(generated_text)
        parse_mode = "last_numeric_fallback" if prediction.ok else "failed"
    return {
        "record_id": record["record_id"],
        "dataset": record["dataset"],
        "source_index": int(record["source_index"]),
        "question_sha256": record["question_sha256"],
        "prompt_version": PROMPT_VERSION,
        "raw_output": generated_text,
        "strict_parse_status": strict.status,
        "parse_mode": parse_mode,
        "parse_status": prediction.status,
        "parsed_prediction": str(prediction.value) if prediction.value is not None else None,
        "gold_value": str(gold),
        "numeric_correct": bool(prediction.ok and prediction.value == gold),
    }


def build_ood_prompt(dataset: str, row: dict[str, Any]) -> str:
    return build_evaluation_prompt(source_question(dataset, row))
