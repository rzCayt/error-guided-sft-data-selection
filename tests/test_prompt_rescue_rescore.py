import importlib.util
from pathlib import Path

from eg_sft.eval.parser import parse_numeric_answer_marker_v2

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rescore_prompt_rescue_outputs.py"
_SPEC = importlib.util.spec_from_file_location("rescore_prompt_rescue_outputs", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_parser_v2_recovers_final_value_before_tail_fragment() -> None:
    text = "The final value is 30.6.\n\nProblem: A number is 10% larger."

    assert parse_numeric_answer_marker_v2(text) == (30.6, "final_value_marker")


def test_parser_v2_keeps_last_number_fallback() -> None:
    assert parse_numeric_answer_marker_v2("7 * 9 = 63") == (63.0, "last_number_fallback")


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
