from __future__ import annotations

import copy
from pathlib import Path

import pytest

from eg_sft.data.public_gsm8k import sha256_text
from eg_sft.training.reasoning import (
    RewardTrace, adapter_fingerprint, build_grpo_trainer, build_sft_trainer, correctness_reward,
    format_reward, reasoning_partitions, reasoning_rows,
)


def test_reward_and_evaluation_share_strict_final_answer_semantics() -> None:
    outputs = ["Work\nFinal answer: 1,200", "Final answer: -2.50", "Final answer: 3\nextra",
               "Final answer: 3\nFinal answer: 3", "3", "Final answer: 4"]
    assert correctness_reward(outputs, ["1200", "-2.5", "3", "3", "3", "3"]) == [1, 1, 0, 0, 0, 0]
    assert format_reward(outputs) == [1, 1, 0, 0, 0, 1]
    assert correctness_reward([[{"role": "assistant", "content": "Final answer: 2"}]], ["2"]) == [1]
    with pytest.raises(ValueError, match="counts differ"):
        correctness_reward(outputs, ["1"])
    with pytest.raises(ValueError, match="finite"):
        correctness_reward(["Final answer: 1"], ["NaN"])
    with pytest.raises(TypeError, match="invalid reward input"):
        format_reward([{"content": "Final answer: 1"}])


def test_partition_is_deterministic_disjoint_and_rejects_duplicate_questions() -> None:
    records = [{"record_id": str(i), "question_sha256": str(i), "protocol_split": "development"}
               for i in range(8)]
    left, right = reasoning_partitions(records, source_split="development", train_count=4,
                                       validation_count=4, seed=17)
    assert {row["record_id"] for row in left}.isdisjoint(row["record_id"] for row in right)
    assert (left, right) == reasoning_partitions(records, source_split="development", train_count=4,
                                                validation_count=4, seed=17)
    records[1]["question_sha256"] = records[0]["question_sha256"]
    with pytest.raises(ValueError, match="Duplicate"):
        reasoning_partitions(records, source_split="development", train_count=4,
                              validation_count=4, seed=17)


def test_source_hash_mismatch_blocks_training() -> None:
    row = {"question": "What is 1 + 1?", "answer": "Add them.\n#### 2"}
    record = {"record_id": "fixture", "source_split": "train", "source_index": 0,
              "question_sha256": sha256_text(row["question"]),
              "answer_sha256": sha256_text(row["answer"])}
    assert reasoning_rows([record], [row])[0]["answer"] == "2"
    with pytest.raises(ValueError, match="question hash"):
        reasoning_rows([record], [{**row, "question": "changed"}])


def test_reward_trace_exposes_uninformative_groups(tmp_path: Path) -> None:
    trace = RewardTrace(tmp_path / "rollouts.jsonl", group_size=4)
    outputs = ["Final answer: 2", "Final answer: 1", "Final answer: 2", "invalid"]
    assert trace.reward(outputs, ["2"] * 4, ["a"] * 4) == [1, 0, 1, 0]
    assert (trace.total_groups, trace.informative_groups) == (1, 1)
    trace.reward(["invalid"] * 4, ["2"] * 4, ["b"] * 4)
    assert (trace.total_groups, trace.informative_groups) == (2, 1)
    with pytest.raises(ValueError, match="mixes prompts"):
        trace.reward(outputs, ["2"] * 4, ["a", "b", "a", "b"])


def tiny_model_and_tokenizer():
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(17)
    tokens = ["<pad>", "<eos>", "<unk>", "Question:", "Solution:", "Final", "answer:",
              "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    backend = Tokenizer(WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                         eos_token="<eos>", unk_token="<unk>", padding_side="left")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokens), hidden_size=32,
                                       intermediate_size=64, num_hidden_layers=1,
                                       num_attention_heads=2, num_key_value_heads=2,
                                       max_position_embeddings=128, pad_token_id=0, eos_token_id=1))
    base.config.use_cache = False
    clean_base = copy.deepcopy(base)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    return model, clean_base, tokenizer


def test_offline_real_sft_then_grpo_updates_and_adapter_reload(tmp_path: Path) -> None:
    """Real library steps on random tiny weights: API evidence, not math results."""
    import torch
    from peft import PeftModel

    from eg_sft.training.response_only import tokenize_response_only

    torch.set_num_threads(1)
    model, clean_base, tokenizer = tiny_model_and_tokenizer()
    before = adapter_fingerprint(model)
    examples = [tokenize_response_only(tokenizer, prompt="Question: one Solution: ",
                                       response="two Final answer: two", max_length=32)] * 4
    config = {"max_steps": 2, "batch_size": 1, "gradient_accumulation": 1,
              "learning_rate": 0.01, "seed": 17, "save_steps": 100, "max_wall_seconds": 60}
    sft = build_sft_trainer(model=model, tokenizer=tokenizer, examples=examples,
                            output_dir=tmp_path / "sft", config=config, use_cpu=True)
    sft.train()
    sft_fingerprint = adapter_fingerprint(model)
    assert before != sft_fingerprint and sft.state.global_step == 2
    model.save_pretrained(tmp_path / "adapter")
    continued = PeftModel.from_pretrained(clean_base, tmp_path / "adapter", is_trainable=True)
    assert adapter_fingerprint(continued) == sft_fingerprint

    # Diverse synthetic rewards qualify library optimization only, not correctness learning.
    def synthetic_reward(completions, **kwargs):
        return [float(sum(ord(character) for character in text) % 101) for text in completions]

    grpo_config = {**config, "gradient_accumulation": 4, "learning_rate": 0.001,
                   "num_generations": 4, "max_completion_length": 6, "temperature": 1.0,
                   "reward_weights": [1.0, 0.0]}
    rows = [{"record_id": str(i), "prompt": "Question: one Solution: ", "answer": "2"}
            for i in range(4)]
    grpo = build_grpo_trainer(model=continued, tokenizer=tokenizer, rows=rows,
                              output_dir=tmp_path / "grpo", config=grpo_config,
                              reward_funcs=[synthetic_reward, format_reward], use_cpu=True)
    grpo.train()
    assert grpo.state.global_step == 2
    assert adapter_fingerprint(continued) != sft_fingerprint
    assert any("reward" in row for row in grpo.state.log_history)
