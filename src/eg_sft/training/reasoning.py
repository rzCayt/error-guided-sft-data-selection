"""Small, auditable SFT/GRPO components using the existing arithmetic protocol."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row
from eg_sft.gsm8k.parser import parse_generated_answer, parse_gold_answer
from eg_sft.training.overfit import format_decimal, gsm8k_training_text


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if (isinstance(completion, list) and len(completion) == 1
            and isinstance(completion[0], dict) and completion[0].get("role") == "assistant"
            and isinstance(completion[0].get("content"), str)):
        return completion[0]["content"]
    raise TypeError("Expected text or one assistant message; invalid reward input")


def correctness_reward(completions: list[Any], answer: list[str], **kwargs: Any) -> list[float]:
    if len(completions) != len(answer):
        raise ValueError("Completion and ground-truth counts differ")
    rewards = []
    for completion, target in zip(completions, answer, strict=True):
        gold = Decimal(target)
        if not gold.is_finite():
            raise ValueError("Ground truth must be finite")
        parsed = parse_generated_answer(completion_text(completion))
        rewards.append(float(parsed.ok and parsed.value == gold))
    return rewards


def format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    return [float(parse_generated_answer(completion_text(item)).ok) for item in completions]


def reasoning_partitions(
    records: list[dict[str, Any]], *, source_split: str, train_count: int,
    validation_count: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("Both partitions must be non-empty")
    ids = [record["record_id"] for record in records]
    hashes = [record["question_sha256"] for record in records]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate manifest IDs or question hashes")
    selected = sorted((row for row in records if row["protocol_split"] == source_split),
                      key=lambda row: row["record_id"])
    if len(selected) < train_count + validation_count:
        raise ValueError("Insufficient rows in the declared source partition")
    random.Random(seed).shuffle(selected)
    return selected[:train_count], selected[train_count:train_count + validation_count]


def reasoning_rows(records: list[dict[str, Any]], source: Any) -> list[dict[str, str]]:
    rows = []
    for record in records:
        if record["source_split"] != "train":
            raise ValueError("Only pinned source training rows may enter this feasibility study")
        row = source[int(record["source_index"])]
        validate_gsm8k_source_row(record, row)
        prompt, response = gsm8k_training_text(row["question"], row["answer"])
        gold = parse_gold_answer(row["answer"])
        if gold.value is None:
            raise ValueError("Invalid source gold answer")
        rows.append({"record_id": record["record_id"], "prompt": prompt,
                     "response": response, "answer": format_decimal(gold.value)})
    return rows


def adapter_fingerprint(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    parameters = [(name, value) for name, value in model.named_parameters() if "lora_" in name]
    if not parameters:
        raise ValueError("No LoRA parameters found")
    for name, value in sorted(parameters):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def training_guard(max_wall_seconds: float) -> Any:
    """Fail on non-finite training and elapsed wall budget; preserve checkpoints."""
    import torch
    from transformers import TrainerCallback

    from eg_sft.training.lora_audit import audit_lora_gradients

    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("Positive wall-time budget required")

    class Guard(TrainerCallback):
        def __init__(self) -> None:
            self.started = time.monotonic()

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if time.monotonic() - self.started > max_wall_seconds:
                raise TimeoutError("Training wall-time limit reached")

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any,
                                  model: Any, **kwargs: Any) -> None:
            audit_lora_gradients(model)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient: {name}")

        def on_log(self, args: Any, state: Any, control: Any,
                   logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            for key, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite training metric: {key}")

    return Guard()


def build_sft_trainer(*, model: Any, tokenizer: Any, examples: list[dict[str, Any]],
                      output_dir: Path, config: dict[str, Any], use_cpu: bool = False) -> Any:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    from eg_sft.training.response_only import ResponseOnlyCollator

    args = TrainingArguments(
        output_dir=str(output_dir), max_steps=config["max_steps"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"], warmup_ratio=0.0, lr_scheduler_type="constant",
        weight_decay=0.0, max_grad_norm=1.0, seed=config["seed"], data_seed=config["seed"],
        bf16=not use_cpu, use_cpu=use_cpu, gradient_checkpointing=not use_cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="steps", save_steps=config["save_steps"],
        save_total_limit=None, report_to="none", disable_tqdm=True,
        remove_unused_columns=False, dataloader_pin_memory=not use_cpu,
    )
    return Trainer(model=model, args=args, processing_class=tokenizer,
                   train_dataset=Dataset.from_list(examples),
                   data_collator=ResponseOnlyCollator(int(tokenizer.pad_token_id)),
                   callbacks=[training_guard(config["max_wall_seconds"])])


def build_grpo_trainer(*, model: Any, tokenizer: Any, rows: list[dict[str, Any]],
                       output_dir: Path, config: dict[str, Any], reward_funcs: list[Any],
                       use_cpu: bool = False) -> Any:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    group_size = config["num_generations"]
    effective = config["batch_size"] * config["gradient_accumulation"]
    if group_size < 2 or effective % group_size:
        raise ValueError("Effective generation batch must be divisible by group size >= 2")
    args = GRPOConfig(
        output_dir=str(output_dir), max_steps=config["max_steps"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        generation_batch_size=effective, num_generations=group_size,
        learning_rate=config["learning_rate"], lr_scheduler_type="constant", warmup_ratio=0.0,
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"], beta=0.0, num_iterations=1,
        loss_type="grpo", scale_rewards="group", reward_weights=config["reward_weights"],
        mask_truncated_completions=False, use_vllm=False,
        bf16=not use_cpu, use_cpu=use_cpu, gradient_checkpointing=not use_cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="steps", save_steps=config["save_steps"],
        save_total_limit=None, report_to="none", disable_tqdm=True,
        dataloader_pin_memory=not use_cpu, seed=config["seed"], data_seed=config["seed"],
    )
    dataset_rows = [{key: row[key] for key in ("record_id", "prompt", "answer")} for row in rows]
    return GRPOTrainer(model=model, args=args, processing_class=tokenizer,
                       train_dataset=Dataset.from_list(dataset_rows), reward_funcs=reward_funcs,
                       callbacks=[training_guard(config["max_wall_seconds"])])


class RewardTrace:
    """Record actual rollouts and count groups that carry a correctness signal."""

    def __init__(self, path: Path, group_size: int) -> None:
        self.path = path
        self.group_size = group_size
        self.total_groups = 0
        self.informative_groups = 0

    def reward(self, completions: list[Any], answer: list[str], record_id: list[str],
               **kwargs: Any) -> list[float]:
        rewards = correctness_reward(completions, answer)
        if len(rewards) != len(record_id) or len(rewards) % self.group_size:
            raise ValueError("Rollouts do not cover complete prompt groups")
        for start in range(0, len(rewards), self.group_size):
            ids = record_id[start:start + self.group_size]
            targets = answer[start:start + self.group_size]
            if len(set(ids)) != 1 or len(set(targets)) != 1:
                raise ValueError("A GRPO group mixes prompts or gold answers")
            group = rewards[start:start + self.group_size]
            self.total_groups += 1
            self.informative_groups += int(len(set(group)) > 1)
        with self.path.open("a", encoding="utf-8") as handle:
            for rid, target, completion, reward in zip(record_id, answer, completions, rewards,
                                                       strict=True):
                text = completion_text(completion)
                handle.write(json.dumps({"record_id": rid, "gold_answer": target,
                                         "completion": text, "correctness_reward": reward,
                                         "parse_status": parse_generated_answer(text).status}) + "\n")
        return rewards
