from __future__ import annotations

import math

from eg_sft.analysis.behavior_composition import (
    exact_four_vs_four_pvalue,
    extract_response_features,
    spearman,
    standardized_mean_difference,
)


def test_extracts_frozen_terminal_categories() -> None:
    final = extract_response_features("Reasoning\n\nFinal answer: 42")
    assert final["terminal_answer_marker_family"] == "final_answer_colon"
    assert final["terminal_answer_marker_present"] is True
    assert final["terminal_structure_category"] == "natural_language_sentence"
    assert final["nonempty_line_count"] == 2

    formula = extract_response_features("Work\n-12.5%")
    assert formula["terminal_structure_category"] == "standalone_numeric_or_formula"
    assert formula["standalone_numeric_or_formula"] is True

    listed = extract_response_features("Answer\n1. 42 apples")
    assert listed["terminal_structure_category"] == "list_item_terminal"


def test_exact_statistics_are_deterministic() -> None:
    values = [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3]
    labels = [False, False, False, False, True, True, True, True]
    assert exact_four_vs_four_pvalue(values, labels) == 2 / 70
    assert math.isclose(spearman(values, values), 1.0)
    assert standardized_mean_difference(values[4:], values[:4]) > 1.0
