"""Reusable runtime helpers for the validated cloud-v2 training calibration."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM

from eg_sft.training.b500 import file_sha256
from eg_sft.training.effective_batch import shifted_response_loss_sums


def build_calibration_model(
    *,
    protocol: dict[str, Any],
    training: dict[str, Any],
    attention_implementation: str,
    gradient_checkpointing: bool,
    device: torch.device,
) -> torch.nn.Module:
    """Build the same BF16 LoRA model for every calibration profile."""

    model_config = protocol["model"]
    model = AutoModelForCausalLM.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attention_implementation,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()
    return get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(training["lora"]["r"]),
            lora_alpha=int(training["lora"]["alpha"]),
            lora_dropout=float(training["lora"]["dropout"]),
            target_modules=training["lora"]["target_modules"],
            bias=training["lora"]["bias"],
        ),
    ).to(device)


def calibration_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    rng_state: dict[str, Any],
    next_micro_batch_index: int,
    optimizer_steps: int,
    supervised_tokens_seen: int,
    response_loss_sum_seen: float,
    compute_seconds_completed: float,
    wall_seconds_completed: float,
    max_peak_memory_bytes_seen: int,
    temperature_sample_count: int,
) -> dict[str, Any]:
    """Serialize only optimizer-boundary state so resume cannot split an update."""

    return {
        "adapter_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in get_peft_model_state_dict(model).items()
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": rng_state,
        "next_micro_batch_index": next_micro_batch_index,
        "optimizer_steps": optimizer_steps,
        "supervised_tokens_seen": supervised_tokens_seen,
        "response_loss_sum_seen": response_loss_sum_seen,
        "compute_seconds_completed": compute_seconds_completed,
        "wall_seconds_completed": wall_seconds_completed,
        "max_peak_memory_bytes_seen": max_peak_memory_bytes_seen,
        "temperature_sample_count": temperature_sample_count,
        "pending_micro_batches": 0,
        "pending_response_tokens": 0,
    }


def mean_response_token_loss(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> float:
    """Compute one response-token-normalized loss without changing parameters."""

    model.eval()
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss_sums, token_counts = shifted_response_loss_sums(
            logits=outputs.logits,
            labels=batch["labels"],
        )
    return float(loss_sums.sum().item() / int(token_counts.sum().item()))


def validate_data_bindings(*, repo_root: Any, config: dict[str, Any]) -> Any:
    """Validate both frozen data hashes without importing the formal runner."""

    from pathlib import Path

    from eg_sft.experiment.cloud_v2_calibration import repository_path

    data = config["data_manifest"]
    directory = repository_path(Path(repo_root), str(data["directory"]), label="data manifest")
    for filename, expected in data["required_files"].items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"frozen data artifact changed: {filename}")
    return directory


def expected_temperature_sample_count(*, optimizer_steps_planned: int) -> int:
    """One start sample plus exactly one sample per optimizer boundary."""

    if optimizer_steps_planned <= 0:
        raise ValueError("optimizer_steps_planned must be positive")
    return optimizer_steps_planned + 1
