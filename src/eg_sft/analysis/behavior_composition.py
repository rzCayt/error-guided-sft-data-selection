"""Frozen feature extraction for selected-response composition audits."""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Any


_FINAL_ANSWER = re.compile(r"^\s*final\s+answer\s*:\s*\S.*$", re.IGNORECASE)
_HASH_ANSWER = re.compile(r"^\s*####\s+\S.*$")
_ANSWER = re.compile(r"^\s*(?:answer|ans)\s*:\s*\S.*$", re.IGNORECASE)
_DECLARATIVE = re.compile(r"^\s*(?:therefore|thus|hence|so)\b.*$", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s*(?:[-*+] |\d+[.)] )\S.*$")
_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_BULLET = re.compile(r"^\s*(?:[-*+] |\d+[.)] )\S")
_LETTER_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_FORMULA_ALLOWED = re.compile(
    r"^[\d\s.,+\-*/%^=()\[\]{}<>:$€£¥\\×÷·]+$",
    re.UNICODE,
)


def normalize_for_features(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n").replace(
        "\r", "\n"
    )


def terminal_answer_marker_family(terminal_line: str) -> str:
    for name, pattern in (
        ("final_answer_colon", _FINAL_ANSWER),
        ("hash_answer", _HASH_ANSWER),
        ("answer_colon", _ANSWER),
        ("declarative_conclusion", _DECLARATIVE),
    ):
        if pattern.match(terminal_line):
            return name
    return "no_terminal_answer_marker"


def _terminal_inside_code_fence(lines: Sequence[str]) -> bool:
    if not lines:
        return False
    if lines[-1].strip().startswith("```"):
        return True
    return sum(line.count("```") for line in lines[:-1]) % 2 == 1


def terminal_structure_category(lines: Sequence[str]) -> str:
    if not lines:
        return "other"
    terminal = lines[-1]
    if _terminal_inside_code_fence(lines):
        return "code_fence_terminal"
    if _LIST_ITEM.match(terminal):
        return "list_item_terminal"
    if any(character.isdigit() for character in terminal) and _FORMULA_ALLOWED.fullmatch(
        terminal
    ):
        return "standalone_numeric_or_formula"
    if len(_LETTER_TOKEN.findall(terminal)) >= 2:
        return "natural_language_sentence"
    return "other"


def extract_response_features(text: str) -> dict[str, Any]:
    normalized = normalize_for_features(text)
    nonempty_lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    terminal = nonempty_lines[-1] if nonempty_lines else ""
    marker_family = terminal_answer_marker_family(terminal)
    structure = terminal_structure_category(nonempty_lines)
    line_count = len(nonempty_lines)
    denominator = max(1, line_count)
    return {
        "terminal_answer_marker_family": marker_family,
        "terminal_answer_marker_present": marker_family
        != "no_terminal_answer_marker",
        "terminal_structure_category": structure,
        "standalone_numeric_or_formula": structure
        == "standalone_numeric_or_formula",
        "nonempty_line_count": line_count,
        "terminal_character_length": len(terminal),
        "markdown_heading_density": sum(bool(_HEADING.match(line)) for line in nonempty_lines)
        / denominator,
        "bullet_line_density": sum(bool(_BULLET.match(line)) for line in nonempty_lines)
        / denominator,
        "code_fence_density": normalized.count("```") / denominator,
    }


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample variance requires at least two values")
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def standardized_mean_difference(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    pooled = math.sqrt(
        ((len(group_a) - 1) * sample_variance(group_a)
         + (len(group_b) - 1) * sample_variance(group_b))
        / (len(group_a) + len(group_b) - 2)
    )
    difference = mean(group_a) - mean(group_b)
    if pooled == 0:
        return 0.0 if difference == 0 else math.copysign(math.inf, difference)
    return difference / pooled


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("correlation inputs must have equal length of at least two")
    center_a = mean(values_a)
    center_b = mean(values_b)
    numerator = sum(
        (value_a - center_a) * (value_b - center_b)
        for value_a, value_b in zip(values_a, values_b, strict=True)
    )
    denominator = math.sqrt(
        sum((value - center_a) ** 2 for value in values_a)
        * sum((value - center_b) ** 2 for value in values_b)
    )
    return 0.0 if denominator == 0 else numerator / denominator


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    return pearson(_average_ranks(values_a), _average_ranks(values_b))


def exact_four_vs_four_pvalue(values: Sequence[float], observed_labels: Sequence[bool]) -> float:
    if len(values) != 8 or len(observed_labels) != 8 or sum(observed_labels) != 4:
        raise ValueError("exact test requires eight values with four labels per group")
    observed_a = [value for value, label in zip(values, observed_labels, strict=True) if label]
    observed_b = [value for value, label in zip(values, observed_labels, strict=True) if not label]
    observed = abs(mean(observed_a) - mean(observed_b))
    exceed = 0
    total = 0
    for indexes in itertools.combinations(range(8), 4):
        selected = set(indexes)
        group_a = [value for index, value in enumerate(values) if index in selected]
        group_b = [value for index, value in enumerate(values) if index not in selected]
        exceed += abs(mean(group_a) - mean(group_b)) >= observed - 1e-15
        total += 1
    return exceed / total
