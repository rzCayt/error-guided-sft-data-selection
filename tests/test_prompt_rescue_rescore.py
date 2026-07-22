import importlib.util
import sys
from pathlib import Path

from eg_sft.eval.parser import (
    parse_numeric_answer_marker_v2,
    parse_numeric_final_marker_only_v4,
    parse_numeric_strict_final_answer_v3,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rescore_prompt_rescue_outputs.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("rescore_prompt_rescue_outputs", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_parser_v2_recovers_final_value_before_tail_fragment() -> None:
    text = "The final value is 30.6.\n\nProblem: A number is 10% larger."

    assert parse_numeric_answer_marker_v2(text) == (30.6, "final_value_marker")


def test_parser_v2_keeps_last_number_fallback() -> None:
    assert parse_numeric_answer_marker_v2("7 * 9 = 63") == (63.0, "last_number_fallback")


def test_parser_v3_rejects_formula_without_final_marker() -> None:
    assert parse_numeric_strict_final_answer_v3("0.7*23 + 0.3*90 = 43.1") == (
        None,
        "non_strict_numeric_output",
    )


def test_parser_v3_accepts_single_numeric_and_final_marker() -> None:
    assert parse_numeric_strict_final_answer_v3("Final numeric answer: 43.1") == (
        43.1,
        "final_answer_marker",
    )
    assert parse_numeric_strict_final_answer_v3(" 43.1 ") == (43.1, "single_numeric_output")


def test_parser_v3_boundary_cases_are_frozen() -> None:
    assert parse_numeric_strict_final_answer_v3("Final answer: 1 + 2 = 3") == (
        None,
        "non_strict_numeric_output",
    )
    assert parse_numeric_strict_final_answer_v3(r"Final answer: \boxed{72.87}") == (
        72.87,
        "final_answer_marker",
    )
    assert parse_numeric_strict_final_answer_v3("Final value is **61.034**") == (
        61.034,
        "final_value_marker",
    )
    assert parse_numeric_strict_final_answer_v3(r"\boxed{72.87}") == (
        72.87,
        "single_numeric_output",
    )


def test_final_marker_only_parser_accepts_explicit_final_markers() -> None:
    assert parse_numeric_final_marker_only_v4("Final answer: 72.87") == (
        72.87,
        "final_answer_marker",
    )
    assert parse_numeric_final_marker_only_v4("Final numeric answer: 72.87") == (
        72.87,
        "final_numeric_answer_marker",
    )
    assert parse_numeric_final_marker_only_v4("Final value is **61.034**") == (
        61.034,
        "final_value_marker",
    )


def test_final_marker_only_parser_rejects_single_number_without_marker() -> None:
    assert parse_numeric_final_marker_only_v4("72.87") == (
        None,
        "missing_final_marker_numeric_output",
    )


def test_final_marker_only_parser_rejects_non_strict_marker_payload() -> None:
    assert parse_numeric_final_marker_only_v4("Final answer: 1 + 2 = 3") == (
        None,
        "non_strict_final_marker_payload",
    )


def test_rescore_row_records_flip() -> None:
    row = {
        "id": "dev_diagnostic-0000",
        "model": "Qwen/Qwen2.5-0.5B",
        "prompt_variant": "current_direct",
        "answer": 30.6,
        "parsed_prediction": 10.0,
        "parser_mode": "last_number_fallback",
        "numeric_accuracy": False,
        "raw_continuation": "The final value is 30.6.\nProblem: A number is 10%",
    }

    rescored = _MODULE.rescore_row(row)

    assert rescored["parser_v2_prediction"] == 30.6
    assert rescored["parser_v2_mode"] == "final_value_marker"
    assert rescored["parser_flip"] == "incorrect_to_correct"


def test_rescore_summary_counts_v1_v2_deltas() -> None:
    rows = [
        {
            "model": "Qwen/Qwen2.5-0.5B",
            "prompt_variant": "current_direct",
            "parser_v1_correct": False,
            "parser_v2_correct": True,
            "parser_v1_mode": "last_number_fallback",
            "parser_v2_mode": "final_value_marker",
            "parser_v2_numeric_token_count": 3,
            "parser_flip": "incorrect_to_correct",
        },
        {
            "model": "Qwen/Qwen2.5-0.5B",
            "prompt_variant": "current_direct",
            "parser_v1_correct": True,
            "parser_v2_correct": True,
            "parser_v1_mode": "last_number_fallback",
            "parser_v2_mode": "last_number_fallback",
            "parser_v2_numeric_token_count": 1,
            "parser_flip": "correct_stable",
        },
    ]

    summary = _MODULE.summarize(rows)

    assert summary[0]["parser_v1_accuracy"] == 0.5
    assert summary[0]["parser_v2_accuracy"] == 1.0
    assert summary[0]["accuracy_delta_v2_minus_v1"] == 0.5
    assert summary[0]["incorrect_to_correct_flips"] == 1
