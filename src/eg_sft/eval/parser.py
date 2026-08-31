from __future__ import annotations

import re

NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
STRICT_NUMBER = r"[-+]?(?:\d*\.\d+|\d+)"
FINAL_ANSWER_RE = re.compile(
    r"final\s+(?:numeric\s+)?answer\s*(?:is|=|:)\s*"
    r"([-+]?(?:\d*\.\d+|\d+))",
    re.IGNORECASE,
)
FINAL_VALUE_RE = re.compile(
    r"final\s+value\s*(?:is|=|:)\s*"
    r"([-+]?(?:\d*\.\d+|\d+))",
    re.IGNORECASE,
)
FINAL_ANSWER_TAIL_RE = re.compile(
    r"final\s+(?:numeric\s+)?answer\s*(?:is|=|:)\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
FINAL_VALUE_TAIL_RE = re.compile(
    r"final\s+value\s*(?:is|=|:)\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
FINAL_NUMERIC_ANSWER_TAIL_RE = re.compile(
    r"final\s+numeric\s+answer\s*(?:is|=|:)\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
SINGLE_NUMERIC_OUTPUT_RE = re.compile(
    rf"^\s*(?:\$+\s*)?(?:\\boxed\{{\s*)?(?:\*\*)?\s*({STRICT_NUMBER})\s*(?:\*\*)?(?:\s*\}})?(?:\s*\$+)?\s*\.?\s*$"
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


def parse_numeric_answer_marker_v2(text: str) -> tuple[float | None, str]:
    cleaned = text.replace(",", "")
    final_answer_matches = FINAL_ANSWER_RE.findall(cleaned)
    if final_answer_matches:
        return float(final_answer_matches[-1]), "final_answer_marker"

    final_value_matches = FINAL_VALUE_RE.findall(cleaned)
    if final_value_matches:
        return float(final_value_matches[-1]), "final_value_marker"

    fallback = parse_numeric_last_number(cleaned)
    if fallback is not None:
        return fallback, "last_number_fallback"
    return None, "parse_failure"


def _first_marker_payload(tail: str) -> str:
    for line in tail.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _parse_strict_single_numeric_payload(payload: str) -> float | None:
    match = SINGLE_NUMERIC_OUTPUT_RE.match(payload)
    if not match:
        return None
    return float(match.group(1))


def parse_numeric_strict_final_answer_v3(text: str) -> tuple[float | None, str]:
    cleaned = text.replace(",", "").strip()
    final_answer_matches = FINAL_ANSWER_TAIL_RE.findall(cleaned)
    if final_answer_matches:
        value = _parse_strict_single_numeric_payload(_first_marker_payload(final_answer_matches[-1]))
        if value is not None:
            return value, "final_answer_marker"
        return None, "non_strict_numeric_output"

    final_value_matches = FINAL_VALUE_TAIL_RE.findall(cleaned)
    if final_value_matches:
        value = _parse_strict_single_numeric_payload(_first_marker_payload(final_value_matches[-1]))
        if value is not None:
            return value, "final_value_marker"
        return None, "non_strict_numeric_output"

    single_numeric = _parse_strict_single_numeric_payload(cleaned)
    if single_numeric is not None:
        return single_numeric, "single_numeric_output"

    if NUMBER_RE.search(cleaned):
        return None, "non_strict_numeric_output"
    return None, "parse_failure"


def parse_numeric_final_marker_only_v4(text: str) -> tuple[float | None, str]:
    """Accept only an explicit final-marker line with one strict numeric payload."""

    cleaned = text.replace(",", "").strip()
    marker_patterns = [
        ("final_numeric_answer_marker", FINAL_NUMERIC_ANSWER_TAIL_RE),
        ("final_answer_marker", FINAL_ANSWER_TAIL_RE),
        ("final_value_marker", FINAL_VALUE_TAIL_RE),
    ]
    marker_seen = False
    for mode, pattern in marker_patterns:
        matches = pattern.findall(cleaned)
        if not matches:
            continue
        marker_seen = True
        value = _parse_strict_single_numeric_payload(_first_marker_payload(matches[-1]))
        if value is not None:
            return value, mode

    if marker_seen:
        return None, "non_strict_final_marker_payload"
    if NUMBER_RE.search(cleaned):
        return None, "missing_final_marker_numeric_output"
    return None, "parse_failure"


def parse_numeric(text: str) -> float | None:
    return parse_numeric_last_number(text)
