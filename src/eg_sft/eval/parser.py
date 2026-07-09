from __future__ import annotations

import re

NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
FINAL_ANSWER_RE = re.compile(
    r"final\s+(?:numeric\s+)?answer\s*(?:is|=|:)\s*"
    r"([-+]?(?:\d*\.\d+|\d+))",
    re.IGNORECASE,
)


def parse_numeric_last_number(text: str) -> float | None:
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])


def parse_numeric_final_answer(text: str) -> float | None:
    matches = FINAL_ANSWER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])


def parse_numeric_final_answer_or_last_number(text: str) -> float | None:
    final_answer = parse_numeric_final_answer(text)
    if final_answer is not None:
        return final_answer
    return parse_numeric_last_number(text)


def parse_numeric_final_answer_or_last_number_with_mode(text: str) -> tuple[float | None, str]:
    final_answer = parse_numeric_final_answer(text)
    if final_answer is not None:
        return final_answer, "final_answer_marker"
    fallback = parse_numeric_last_number(text)
    if fallback is not None:
        return fallback, "last_number_fallback"
    return None, "parse_failure"


def parse_numeric(text: str) -> float | None:
    return parse_numeric_last_number(text)
