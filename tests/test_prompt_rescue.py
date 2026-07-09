import importlib.util
from pathlib import Path

from eg_sft.eval.parser import (
    parse_numeric,
    parse_numeric_final_answer,
    parse_numeric_final_answer_or_last_number,
    parse_numeric_final_answer_or_last_number_with_mode,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_prompt_rescue_diagnostic.py"
_SPEC = importlib.util.spec_from_file_location("run_prompt_rescue_diagnostic", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_final_answer_parser_prefers_explicit_final_answer() -> None:
    text = "75 - 5% of 75 = 75 - 3.75 = 71.25\nFinal answer: 71.25"

    assert parse_numeric(text) == 71.25
    assert parse_numeric_final_answer(text) == 71.25
    assert parse_numeric_final_answer_or_last_number(text) == 71.25


def test_final_answer_parser_avoids_tail_fragment_when_marker_exists() -> None:
    text = "60 - 20% of 60 = 48. Final answer: 48\nWhat is the answer? 4"

    assert parse_numeric(text) == 4.0
    assert parse_numeric_final_answer_or_last_number(text) == 48.0
    assert parse_numeric_final_answer_or_last_number_with_mode(text) == (
        48.0,
        "final_answer_marker",
    )


def test_prompt_rescue_variants_are_explicit_about_final_answer() -> None:
    prompt = _MODULE.build_prompt("Multiply 7 and 9. What is the product?", "final_answer_only")

    assert "Final answer: <number>" in prompt
    assert "Problem: Multiply 7 and 9" in prompt


def test_prompt_rescue_summary_rows_include_gate_metadata() -> None:
    diagnostics = [
        {
            "prompt_variant": "final_answer_only",
            "parser_mode": "final_answer_marker",
            "numeric_token_count": 1,
            "parse_success": True,
            "numeric_accuracy": True,
        },
        {
            "prompt_variant": "final_answer_only",
            "parser_mode": "last_number_fallback",
            "numeric_token_count": 3,
            "parse_success": True,
            "numeric_accuracy": False,
        },
    ]
    metadata = {
        "model": "Qwen/Qwen2.5-0.5B",
        "model_revision": "abc",
        "tokenizer_revision": "abc",
        "dtype": "torch.float32",
        "device": {"type": "cpu"},
        "seed": 1,
        "prompt_rendering": "plain_completion",
        "generation_config": {"max_new_tokens": 64, "do_sample": False},
        "parser_version": _MODULE.PARSER_VERSION,
        "raw_outputs_path": "results/prompt_rescue/Qwen_Qwen2.5-0.5B/prompt_rescue_outputs.jsonl",
    }

    rows = _MODULE.summary_rows(diagnostics, metadata)

    assert rows[0]["condition"] == "qwen_0_5b_prompt_rescue_gate"
    assert rows[0]["split"] == "dev_diagnostic"
    assert rows[0]["numeric_accuracy"] == 0.5
    assert rows[0]["final_answer_marker_rate"] == 0.5
    assert rows[0]["last_number_fallback_rate"] == 0.5
    assert rows[0]["multi_number_output_rate"] == 0.5
    assert rows[0]["absolute_gain_vs_recorded_base"] == 0.29
    assert rows[0]["parser_version"] == _MODULE.PARSER_VERSION
