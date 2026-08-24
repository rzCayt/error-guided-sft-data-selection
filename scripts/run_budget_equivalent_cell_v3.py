"""Public v3 cell entry with a truthful variable-step training artifact writer."""

from __future__ import annotations

import gc
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from _bootstrap import add_src_to_path

add_src_to_path()

import run_budget_equivalent_cell as implementation  # noqa: E402
from run_b500_formal_resumable import _optimizer_to_device, _write_json_exclusive  # noqa: E402

from eg_sft.experiment.cloud_v2_train_runtime import mean_response_token_loss  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402
from eg_sft.training.token_budget import optimizer_step_token_audit  # noqa: E402


def _save_training_complete_v3(
    *,
    run_dir: Path,
    contract: dict[str, Any],
    recipe: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    fixed_batch: dict[str, torch.Tensor],
    parameter_report: Any,
    token_audit: list[dict[str, Any]],
    state: dict[str, Any],
) -> Path:
    final_dir = run_dir / "training_complete"
    if final_dir.is_dir():
        return final_dir
    step_rows = read_jsonl(run_dir / "optimizer_step_tokens.jsonl")
    step_token_counts = [int(row["response_supervision_tokens"]) for row in step_rows]
    expected_exposure = sum(int(row["supervised_tokens"]) for row in token_audit) * int(
        recipe["training"]["epochs"]
    )
    budget_audit = optimizer_step_token_audit(
        step_token_counts=step_token_counts,
        expected_optimizer_steps=int(recipe["training"]["optimizer_steps"]),
        expected_exposure_tokens=expected_exposure,
        tolerance_fraction=0.005,
    )
    if not budget_audit["exposure_gate_passed"]:
        raise ValueError("Phase 1 response-token exposure gate failed")

    pre_reload_loss = mean_response_token_loss(model=model, batch=fixed_batch)
    attempt = run_dir / f"training_complete_attempt_{uuid.uuid4().hex}"
    attempt.mkdir(parents=False, exist_ok=False)
    adapter_dir = attempt / "adapter"
    tokenizer_dir = attempt / "tokenizer"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    adapter_sha256 = file_sha256(adapter_dir / "adapter_model.safetensors")
    model.to("cpu")
    _optimizer_to_device(optimizer, torch.device("cpu"))
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_base = AutoModelForCausalLM.from_pretrained(
        contract["protocol"]["model"]["repo_id"],
        revision=contract["protocol"]["model"]["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=str(recipe["training"]["attention_implementation"]),
    )
    reloaded = PeftModel.from_pretrained(
        reloaded_base, adapter_dir, is_trainable=False
    ).to("cuda")
    post_reload_loss = mean_response_token_loss(model=reloaded, batch=fixed_batch)
    reload_difference = abs(post_reload_loss - pre_reload_loss)
    if reload_difference > 1e-6:
        raise RuntimeError("Phase 1 adapter reload loss check failed")
    metrics = {
        "status": "PASS",
        "selected_count": len(token_audit),
        "epochs": int(recipe["training"]["epochs"]),
        "micro_batch_size": int(recipe["training"]["micro_batch_size"]),
        "optimizer_step_partition": "balanced_sequence_groups_16_or_15",
        "optimizer_steps_planned": int(recipe["training"]["optimizer_steps"]),
        "optimizer_steps_completed": int(state["optimizer_steps"]),
        "supervised_tokens_seen": int(state["supervised_tokens_seen"]),
        "training_wall_seconds": float(state["wall_seconds_completed"]),
        "peak_training_memory_gib": int(state["max_peak_memory_bytes_seen"]) / 1024**3,
        "trainable_parameters": parameter_report.trainable_parameters,
        "total_parameters": parameter_report.total_parameters,
        "adapter_model_sha256": adapter_sha256,
        "adapter_reload_loss_absolute_difference": reload_difference,
        "accuracy_withheld": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json_exclusive(attempt / "training_metrics.json", metrics)
    _write_json_exclusive(attempt / "token_audit.json", token_audit)
    _write_json_exclusive(attempt / "token_budget_audit.json", budget_audit)
    attempt.rename(final_dir)
    del reloaded, reloaded_base
    gc.collect()
    torch.cuda.empty_cache()
    return final_dir


def main() -> None:
    implementation._save_training_complete = _save_training_complete_v3
    implementation.main()


if __name__ == "__main__":
    main()
