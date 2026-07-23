"""Strict numeric parsing for GSM8K-style evaluation.

The parser deliberately separates two formats:

* dataset gold answers use the canonical ``#### number`` marker;
* model generations must end with ``Final answer: number``.

Model text is never corrected. Ambiguous, missing, or malformed answers are
reported as parse failures so the primary end-to-end accuracy cannot silently
exclude them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
_GOLD_PATTERN = re.compile(rf"####\s*(?P<number>{_NUMBER})\s*$")
_ANY_NUMBER = re.compile(rf"(?P<number>{_NUMBER})")
_GENERATED_LINE = re.compile(
    rf"Final\s+answer\s*:\s*(?P<number>{_NUMBER})\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseResult:
    """A parsed numeric answer plus a machine-readable status."""

    value: Decimal | None
    status: str
    matched_text: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None and self.status == "ok"


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def parse_gold_answer(answer: str) -> ParseResult:
    """Parse the one canonical GSM8K gold marker at the end of an answer."""

    matches = list(_GOLD_PATTERN.finditer(answer))
    if not matches:
        return ParseResult(None, "missing_gold_marker")
    if len(matches) != 1:
        return ParseResult(None, "multiple_gold_markers")

    match = matches[0]
    value = _to_decimal(match.group("number"))
    if value is None:
        return ParseResult(None, "invalid_gold_number", match.group(0))
    return ParseResult(value, "ok", match.group(0))


def parse_generated_answer(text: str) -> ParseResult:
    """Parse exactly one final-line ``Final answer: number`` marker."""

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return ParseResult(None, "empty_output")

    matches = [
        match
        for line in nonempty_lines
        if (match := _GENERATED_LINE.fullmatch(line)) is not None
    ]
    if not matches:
        return ParseResult(None, "missing_final_marker")
    if len(matches) != 1:
        return ParseResult(None, "multiple_final_markers")

    final_match = _GENERATED_LINE.fullmatch(nonempty_lines[-1])
    if final_match is None:
        return ParseResult(None, "marker_not_final")

    value = _to_decimal(final_match.group("number"))
    if value is None:
        return ParseResult(None, "invalid_number", final_match.group(0))
    return ParseResult(value, "ok", final_match.group(0))


def parse_last_numeric_answer(text: str) -> ParseResult:
    """Extract the last numeric token as an explicitly labeled fallback."""

    matches = list(_ANY_NUMBER.finditer(text))
    if not matches:
        return ParseResult(None, "no_numeric_token")
    match = matches[-1]
    value = _to_decimal(match.group("number"))
    if value is None:
        return ParseResult(None, "invalid_last_number", match.group(0))
    return ParseResult(value, "ok", match.group(0))
