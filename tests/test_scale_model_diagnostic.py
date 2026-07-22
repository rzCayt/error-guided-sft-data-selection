import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_scale_model_diagnostic.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("run_scale_model_diagnostic", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.last_enable_thinking = None
        self.last_messages = None

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        self.last_enable_thinking = enable_thinking
        self.last_messages = messages
        return f"thinking={enable_thinking}\n" + "\n".join(
            message["content"] for message in messages
        )


class FakeDtypeCompatModel:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls.append((model_name, kwargs))
        if "dtype" in kwargs:
            raise AttributeError("simulated Qwen3 dtype compatibility failure")
        return {"model": model_name, "kwargs": kwargs}


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_name):
        raise AttributeError("simulated local Qwen3 config coercion failure")


class FakePreTrainedTokenizerFast:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_model_loader_falls_back_after_attribute_error() -> None:
    FakeDtypeCompatModel.calls = []

    result = _MODULE.load_causal_lm_with_dtype_compat(FakeDtypeCompatModel, "local-snapshot")

    assert result["kwargs"] == {"torch_dtype": "auto"}
    assert FakeDtypeCompatModel.calls == [
        ("local-snapshot", {"dtype": "auto"}),
        ("local-snapshot", {"torch_dtype": "auto"}),
    ]


def test_model_loader_can_skip_buggy_qwen3_dtype_path() -> None:
    FakeDtypeCompatModel.calls = []

    result = _MODULE.load_causal_lm_with_dtype_compat(
        FakeDtypeCompatModel,
        "local-qwen3-snapshot",
        prefer_legacy_torch_dtype=True,
    )

    assert result["kwargs"] == {"torch_dtype": "auto"}
    assert FakeDtypeCompatModel.calls == [
        ("local-qwen3-snapshot", {"torch_dtype": "auto"}),
    ]


def test_local_snapshot_tokenizer_fallback_preserves_frozen_fields(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "template",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "model_max_length": 32768,
                "clean_up_tokenization_spaces": False,
                "unrelated_field": "must not be copied",
            }
        ),
        encoding="utf-8",
    )

    tokenizer, method = _MODULE.load_tokenizer_with_local_snapshot_compat(
        FakeAutoTokenizer,
        FakePreTrainedTokenizerFast,
        str(tmp_path),
    )

    assert method.startswith("PreTrainedTokenizerFast")
    assert tokenizer.kwargs["tokenizer_file"] == str(tmp_path / "tokenizer.json")
    assert tokenizer.kwargs["chat_template"] == "template"
    assert tokenizer.kwargs["clean_up_tokenization_spaces"] is False
    assert "unrelated_field" not in tokenizer.kwargs


def test_safe_model_name_keeps_outputs_separate() -> None:
    assert _MODULE.safe_model_name("Qwen/Qwen2.5-1.5B") == "Qwen_Qwen2.5-1.5B"


def test_output_path_can_use_run_name() -> None:
    assert (
        _MODULE.raw_outputs_path(
            "results/model_diagnostic_output_control",
            "Qwen/Qwen2.5-1.5B",
            "qwen2_5_1_5b_max8",
        )
        == "results/model_diagnostic_output_control/qwen2_5_1_5b_max8/scale_model_diagnostic_outputs.jsonl"
    )


def test_qwen3_answer_only_prompt_is_preregistered() -> None:
    prompt = _MODULE.build_prompt("What is 2 + 3?", "qwen3_answer_only")

    assert "only one final numeric value" in prompt
    assert "Do not include reasoning" in prompt
    assert "Problem: What is 2 + 3?" in prompt


def test_qwen3_chat_prompt_requires_tokenizer() -> None:
    try:
        _MODULE.build_prompt("What is 2 + 3?", "qwen3_chat_no_think_answer_only")
    except ValueError as exc:
        assert "requires a tokenizer" in str(exc)
    else:
        raise AssertionError("chat prompt should require tokenizer")


def test_qwen3_strict_single_number_prompt_requires_tokenizer() -> None:
    try:
        _MODULE.build_prompt("What is 2 + 3?", "qwen3_chat_strict_single_number_v1")
    except ValueError as exc:
        assert "requires a tokenizer" in str(exc)
    else:
        raise AssertionError("strict chat prompt should require tokenizer")


def test_qwen3_completion_strict_single_number_prompt_is_plain_completion() -> None:
    prompt = _MODULE.build_prompt("What is 2 + 3?", "qwen3_completion_strict_single_number_v1")

    assert "You are being evaluated by an exact parser" in prompt
    assert "Problem: What is 2 + 3?" in prompt
    assert "Return exactly one numeric value." in prompt
    assert "<|im_start|>" not in prompt


def test_qwen3_chat_cot_final_answer_prompt_disables_thinking() -> None:
    tokenizer = FakeChatTokenizer()

    prompt = _MODULE.build_prompt(
        "What is 2 + 3?",
        "qwen3_chat_cot_final_answer_v1",
        tokenizer,
    )

    assert tokenizer.last_enable_thinking is False
    assert "Final answer: <number>" in prompt
    assert "What is 2 + 3?" in prompt


def test_qwen3_chat_think_final_answer_prompt_enables_thinking() -> None:
    tokenizer = FakeChatTokenizer()

    prompt = _MODULE.build_prompt(
        "What is 2 + 3?",
        "qwen3_chat_think_final_answer_v1",
        tokenizer,
    )

    assert tokenizer.last_enable_thinking is True
    assert "Final answer: <number>" in prompt
    assert _MODULE.qwen3_enable_thinking_for_variant("qwen3_chat_think_final_answer_v1") is True


def test_qwen3_chat_exact_arithmetic_prompt_keeps_thinking_disabled() -> None:
    tokenizer = FakeChatTokenizer()

    prompt = _MODULE.build_prompt(
        "Compute (0.333 * 66) + (0.334 * 36).",
        "qwen3_chat_exact_arithmetic_final_answer_v1",
        tokenizer,
    )

    assert tokenizer.last_enable_thinking is False
    assert "exact decimal arithmetic" in prompt
    assert "do not round intermediate values" in prompt
    assert "Final answer: <number>" in prompt


def test_math_cot_final_answer_prompt_is_preregistered() -> None:
    prompt = _MODULE.build_prompt("What is 2 + 3?", "math_cot_final_answer_v1")

    assert "concise calculation steps" in prompt
    assert "Final answer: <number>" in prompt
    assert "Problem: What is 2 + 3?" in prompt
    assert "5" not in prompt
    assert "<|im_start|>" not in prompt


def test_exact_arithmetic_final_answer_prompt_is_plain_completion() -> None:
    prompt = _MODULE.build_prompt(
        "Compute (0.333 * 66) + (0.334 * 36).",
        "exact_arithmetic_final_answer_v1",
    )

    assert "exact decimal arithmetic" in prompt
    assert "do not round intermediate values" in prompt
    assert "Final answer: <number>" in prompt
    assert "Problem: Compute (0.333 * 66) + (0.334 * 36)." in prompt
    assert "<|im_start|>" not in prompt


def test_infer_generation_termination_detects_max_tokens() -> None:
    result = _MODULE.infer_generation_termination(
        generated_token_ids=[10, 11, 12],
        continuation="1 2 3",
        max_new_tokens=3,
        eos_token_ids=[99],
        stop_at_next_problem=True,
    )

    assert result["generated_token_count"] == 3
    assert result["reached_max_new_tokens"] is True
    assert result["stopping_reason"] == "max_new_tokens"


def test_infer_generation_termination_detects_eos() -> None:
    result = _MODULE.infer_generation_termination(
        generated_token_ids=[10, 99],
        continuation="42",
        max_new_tokens=32,
        eos_token_ids=[99],
        stop_at_next_problem=True,
    )

    assert result["ended_by_eos"] is True
    assert result["reached_max_new_tokens"] is False
    assert result["stopping_reason"] == "eos"


def test_infer_generation_termination_detects_next_problem_stop() -> None:
    result = _MODULE.infer_generation_termination(
        generated_token_ids=[10, 11],
        continuation="42\nProblem: next",
        max_new_tokens=32,
        eos_token_ids=[],
        stop_at_next_problem=True,
    )

    assert result["stop_at_next_problem_triggered"] is True
    assert result["stopping_reason"] == "stop_at_next_problem"


def test_scale_summary_reports_gain_against_0_5b_parser_v2() -> None:
    diagnostics = [
        {
            "parse_success": True,
            "numeric_accuracy": True,
            "parser_mode": "final_value_marker",
        },
        {
            "parse_success": True,
            "numeric_accuracy": False,
            "parser_mode": "last_number_fallback",
        },
    ]
    metadata = {
        "model": "Qwen/Qwen2.5-1.5B",
        "model_revision": "abc",
        "tokenizer_revision": "abc",
        "dtype": "torch.float16",
        "device": {"type": "cuda"},
        "seed": 1,
        "prompt_variant": "current_completion",
        "prompt_template": _MODULE.PROMPT_TEMPLATE,
        "generation_config": {"max_new_tokens": 32, "do_sample": False},
        "parser_version": _MODULE.PARSER_VERSION,
        "raw_outputs_path": "results/model_diagnostic_scale/Qwen_Qwen2.5-1.5B/scale_model_diagnostic_outputs.jsonl",
    }

    rows = _MODULE.summary_rows(diagnostics, metadata)

    assert rows[0]["condition"] == "scale_model_diagnostic"
    assert rows[0]["split"] == "dev_diagnostic"
    assert rows[0]["numeric_accuracy"] == 0.5
    assert rows[0]["gain_vs_qwen2_5_0_5b_parser_v2"] == 0.22
    assert rows[0]["answer_marker_rate"] == 0.5
    assert rows[0]["last_number_fallback_rate"] == 0.5
    assert rows[0]["parser_version"] == _MODULE.PARSER_VERSION
