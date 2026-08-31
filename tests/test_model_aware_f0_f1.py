import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_model_aware_f0_f1.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("run_model_aware_f0_f1", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is True
        assert enable_thinking is False
        prompt_length = len(messages[0]["content"].split())
        prefix = [1, 2] + [3] * prompt_length
        if add_generation_prompt:
            return prefix
        target_length = len(messages[-1]["content"].split())
        return prefix + [4] * target_length + [5]


def row(row_id: str = "candidate-1", prompt: str = "weighted sum") -> dict:
    return {
        "id": row_id,
        "prompt": prompt,
        "rationale": "Compute the products and sum them.",
        "answer": 12.5,
    }


def test_teacher_forcing_masks_only_prefix() -> None:
    serialized = _MODULE.serialize_teacher_forcing(FakeTokenizer(), row())

    prefix = serialized["prefix_token_count"]
    assert serialized["labels"][:prefix] == [-100] * prefix
    assert serialized["labels"][prefix:] == serialized["input_ids"][prefix:]
    assert serialized["target_token_count"] > 0
    assert serialized["total_token_count"] <= 256


def test_teacher_forcing_rejects_over_256_tokens() -> None:
    oversized = row(prompt="word " * 300)

    with pytest.raises(ValueError, match="exceeds 256 tokens"):
        _MODULE.serialize_teacher_forcing(FakeTokenizer(), oversized)


def test_longest_selection_uses_content_hash_not_id() -> None:
    first = row("zzz", "short prompt")
    second = row("aaa", "this is the longest prompt here")
    selected, info, _ = _MODULE.choose_longest(FakeTokenizer(), [first, second])

    assert selected["id"] == "aaa"
    assert info["content_sha256"] == _MODULE.content_hash(second)


def test_parameter_spec_matches_frozen_qwen3_config() -> None:
    class Config:
        num_hidden_layers = 28
        hidden_size = 2048
        num_attention_heads = 16
        num_key_value_heads = 8
        head_dim = 128

    spec = _MODULE.parameter_spec_from_config(Config())

    assert spec["last_layer_index"] == 27
    assert spec["parameters"][0]["shape"] == [2048, 2048]
    assert spec["parameters"][1]["shape"] == [1024, 2048]
    assert spec["total_parameter_count"] == 6_291_456
    assert spec["fp32_gradient_bytes"] == 25_165_824
    assert spec["fp32_gradient_mib"] == 24.0


def test_static_estimate_uses_frozen_reserves() -> None:
    estimate = _MODULE.estimate_peak_mib(3331.66, 24.0)

    assert estimate == pytest.approx(5915.66)
    assert estimate < _MODULE.F0_ESTIMATE_LIMIT_MIB


def test_nvidia_smi_preflight_is_parsed_before_cuda(monkeypatch) -> None:
    class Completed:
        stdout = "8151, 72, 7828\n"

    monkeypatch.setattr(_MODULE.subprocess, "run", lambda *args, **kwargs: Completed())

    memory = _MODULE.query_nvidia_smi_memory_mib()

    assert memory == {"total": 8151.0, "used": 72.0, "free": 7828.0}


def test_frozen_data_filter_selects_qwen3_17_errors_and_8_correct() -> None:
    candidates, errors, correct = _MODULE.load_frozen_rows()

    assert len(candidates) == 125
    assert len(errors) == 17
    assert len(correct) == 8
    assert all(item["task_family"] == "weighted_aggregation" for item in candidates)


def test_correct_query_length_audit_covers_all_8_queries(monkeypatch, tmp_path: Path) -> None:
    import transformers

    correct = [row(f"correct-{index}") for index in range(8)]
    longest = correct[-1]
    info = {
        "content_sha256": _MODULE.content_hash(longest),
        "prefix_token_count": 63,
        "target_token_count": 62,
        "total_token_count": 125,
    }

    class Tokenizer:
        _commit_hash = _MODULE.REVISION

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(_MODULE, "load_frozen_rows", lambda: ([], [], correct))
    monkeypatch.setattr(_MODULE, "choose_longest", lambda tokenizer, rows: (longest, info, [longest["id"]]))
    static_path = tmp_path / "static_estimate.json"
    static_path.write_text("{}", encoding="utf-8")
    static = {
        "longest_candidate": {"total_token_count": 152},
        "longest_error_query": {"total_token_count": 152},
    }

    audited = _MODULE.run_correct_query_length_audit(static, static_path)

    assert audited["correct_query_count"] == 8
    assert audited["longest_correct_query"]["total_token_count"] == 125
    assert audited["correct_query_exceeds_smoked_max"] is False
    assert audited["additional_backward_required"] is False
    assert audited["status"] == "passed"


def test_allowlist_rejects_test_or_unknown_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the frozen allowlist"):
        _MODULE.validate_input_allowlist([tmp_path / "test_id.jsonl"])


class TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2, bias=False)
        self.v_proj = torch.nn.Linear(2, 1, bias=False)
        self.k_proj = torch.nn.Linear(2, 1, bias=False)


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()


class TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = torch.nn.Linear(2, 2, bias=False)


def test_only_selected_qv_parameters_remain_trainable() -> None:
    model = TinyModel()
    selected = [
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
    ]

    trainable = _MODULE.configure_selected_parameters(model, selected)

    assert trainable == selected
    assert [name for name, value in model.named_parameters() if value.requires_grad] == selected


def test_forbidden_research_output_keys_are_rejected() -> None:
    _MODULE.ensure_no_forbidden_output_keys({"status": "passed", "loss_finite": True})

    with pytest.raises(ValueError, match="forbidden output key"):
        _MODULE.ensure_no_forbidden_output_keys({"cosine_score": 0.5})


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    (output / "static_estimate.json").write_text(json.dumps({"status": "passed"}))

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _MODULE.prepare_output_dir(output, "all")
