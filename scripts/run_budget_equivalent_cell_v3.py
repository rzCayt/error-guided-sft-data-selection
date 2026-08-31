"""Public v3 cell entry with a truthful variable-step training artifact writer."""

from __future__ import annotations

import gc
import hashlib
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
from eg_sft.experiment.phase2_v8_snapshot import frozen_model_source  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402
from eg_sft.training.token_budget import (  # noqa: E402
    optimizer_step_token_audit,
    supervision_tokens_per_step,
)


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
    supervision_token_cap = recipe["training"].get("supervision_token_cap")
    token_cap_policy = recipe["training"].get("token_cap_policy")
    if supervision_token_cap is None:
        expected_exposure = sum(
            int(row["supervised_tokens"]) for row in token_audit
        ) * int(recipe["training"]["epochs"])
        tolerance_fraction = 0.005
    else:
        expected_exposure = int(supervision_token_cap)
        expected_per_step = supervision_tokens_per_step(
            supervision_token_cap=expected_exposure,
            optimizer_steps=int(recipe["training"]["optimizer_steps"]),
            policy=str(token_cap_policy),
        )
        if not step_rows:
            raise ValueError("token-cap training has no optimizer-step evidence")
        selected_token_set_sha256 = str(
            step_rows[0].get("selected_token_set_sha256", "")
        )
        boundary_split_occurrence_count = int(
            step_rows[0].get("boundary_split_occurrence_count", -1)
        )
        selected_candidate_id_coverage = int(
            step_rows[0].get("selected_candidate_id_coverage", -1)
        )
        candidate_id_count = int(step_rows[0].get("candidate_id_count", -1))
        occurrence_with_kept_token_count = int(
            step_rows[0].get("occurrence_with_kept_token_count", -1)
        )
        occurrence_count = int(step_rows[0].get("occurrence_count", -1))
        mandatory_coverage_token_count = int(
            step_rows[0].get("mandatory_coverage_token_count", -1)
        )
        cumulative = 0
        for index, row in enumerate(step_rows, start=1):
            kept = int(row.get("kept_response_supervision_tokens", -1))
            candidate = int(row.get("candidate_response_supervision_tokens", -1))
            cumulative += kept
            if (
                kept != expected_per_step
                or int(row["response_supervision_tokens"]) != expected_per_step
                or candidate < kept
                or row.get("token_cap_policy") != token_cap_policy
                or len(str(row.get("token_cap_mask_sha256", ""))) != 64
                or int(row.get("cumulative_response_supervision_tokens", -1))
                != cumulative
                or row.get("selected_token_set_sha256")
                != selected_token_set_sha256
                or row.get("legacy_sequence_step_boundaries_preserved") is not False
                or int(row.get("boundary_split_occurrence_count", -1))
                != boundary_split_occurrence_count
                or int(row.get("selected_candidate_id_coverage", -1))
                != selected_candidate_id_coverage
                or int(row.get("candidate_id_count", -1)) != candidate_id_count
                or int(row.get("occurrence_with_kept_token_count", -1))
                != occurrence_with_kept_token_count
                or int(row.get("occurrence_count", -1)) != occurrence_count
                or int(row.get("mandatory_coverage_token_count", -1))
                != mandatory_coverage_token_count
            ):
                raise ValueError(f"invalid token-cap evidence at optimizer step {index}")
        if (
            len(selected_token_set_sha256) != 64
            or boundary_split_occurrence_count < 0
            or selected_candidate_id_coverage != candidate_id_count
            or candidate_id_count != len(token_audit)
            or occurrence_with_kept_token_count != occurrence_count
            or occurrence_count != len(token_audit) * int(recipe["training"]["epochs"])
            or mandatory_coverage_token_count != occurrence_count
        ):
            raise ValueError("token-cap plan identity evidence is invalid")
        tolerance_fraction = 0.0
    budget_audit = optimizer_step_token_audit(
        step_token_counts=step_token_counts,
        expected_optimizer_steps=int(recipe["training"]["optimizer_steps"]),
        expected_exposure_tokens=expected_exposure,
        tolerance_fraction=tolerance_fraction,
    )
    if supervision_token_cap is not None:
        mask_set_text = "\n".join(
            str(row["token_cap_mask_sha256"]) for row in step_rows
        ) + "\n"
        budget_audit.update(
            {
                "token_cap_policy": token_cap_policy,
                "supervision_token_cap": expected_exposure,
                "supervision_tokens_per_optimizer_step": expected_per_step,
                "optimizer_step_mask_set_sha256": hashlib.sha256(
                    mask_set_text.encode("utf-8")
                ).hexdigest(),
                "candidate_response_supervision_tokens": sum(
                    int(row["candidate_response_supervision_tokens"])
                    for row in step_rows
                ),
                "selected_token_set_sha256": selected_token_set_sha256,
                "legacy_sequence_step_boundaries_preserved": False,
                "boundary_split_occurrence_count": boundary_split_occurrence_count,
                "selected_candidate_id_coverage": selected_candidate_id_coverage,
                "candidate_id_count": candidate_id_count,
                "occurrence_with_kept_token_count": occurrence_with_kept_token_count,
                "occurrence_count": occurrence_count,
                "mandatory_coverage_token_count": mandatory_coverage_token_count,
            }
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
    model_source, source_kwargs = frozen_model_source(
        contract["protocol"]["model"]
    )
    reloaded_base = AutoModelForCausalLM.from_pretrained(
        model_source,
        **source_kwargs,
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
        "optimizer_step_partition": (
            "hash_uniform_global_dose_then_ordered_995_token_blocks"
            if supervision_token_cap is not None
            else "balanced_sequence_groups_16_or_15"
        ),
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
    if supervision_token_cap is not None:
        metrics.update(
            {
                "supervision_token_cap": expected_exposure,
                "token_cap_policy": token_cap_policy,
                "tokens_per_optimizer_step": expected_per_step,
                "selected_token_set_sha256": selected_token_set_sha256,
                "legacy_sequence_step_boundaries_preserved": False,
                "boundary_split_occurrence_count": boundary_split_occurrence_count,
                "selected_candidate_id_coverage": selected_candidate_id_coverage,
                "candidate_id_count": candidate_id_count,
                "occurrence_with_kept_token_count": occurrence_with_kept_token_count,
                "occurrence_count": occurrence_count,
                "mandatory_coverage_token_count": mandatory_coverage_token_count,
            }
        )
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
