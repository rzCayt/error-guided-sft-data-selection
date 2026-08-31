"""Validated 64-example cloud-v2 training calibration entry point."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import (  # noqa: E402
    _append_jsonl,
    _git_commit,
    _gpu_sample,
    _optimizer_to_device,
    _prepare_training_data,
    _read_json,
    _require_clean_git_worktree,
    _restore_rng_state,
    _rng_state,
    _to_device,
    _write_json_exclusive,
)

from eg_sft.experiment.cloud_v2_calibration import (  # noqa: E402
    calibration_run_config,
    read_json_object,
    repository_path,
    resolve_frozen_artifact,
    validate_calibration_config,
)
from eg_sft.experiment.cloud_v2_train_runtime import (  # noqa: E402
    build_calibration_model,
    calibration_checkpoint_payload,
    expected_temperature_sample_count,
    mean_response_token_loss,
    validate_data_bindings,
)
from eg_sft.experiment.formal_runtime import (  # noqa: E402
    deterministic_epoch_orders,
    load_latest_checkpoint,
    write_immutable_checkpoint,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.training.b500 import (  # noqa: E402
    file_sha256,
    selected_id_sha256,
    validate_selection_manifest,
)
from eg_sft.training.effective_batch import (  # noqa: E402
    build_training_micro_batches,
    normalize_gradients_by_token_count,
    optimizer_steps_for_examples,
    shifted_response_loss_sums,
    should_write_checkpoint,
)
from eg_sft.training.lora_audit import audit_lora_gradients, audit_lora_parameters  # noqa: E402
from eg_sft.training.response_only import ResponseOnlyCollator  # noqa: E402


def _create_or_resume_run(
    *,
    output_root: Path,
    run_config: dict[str, Any],
    recipe: dict[str, Any],
    seed: int,
    profile_name: str,
    protocol: dict[str, Any],
    resume_run_dir: Path | None,
) -> tuple[Path, dict[str, Any]]:
    if resume_run_dir is None:
        run_dir, manifest = create_run_manifest(
            output_root=output_root,
            repo_root=ROOT,
            stage=f"cloud_v2_train_calibration_fixed_{profile_name}",
            config=run_config,
            seed=seed,
            command=[str(Path(__file__)), "--profile", profile_name],
            dataset_revisions={
                protocol["datasets"]["gsm8k"]["repo_id"]: protocol["datasets"]["gsm8k"][
                    "revision"
                ],
                protocol["datasets"]["candidate_pool"]["repo_id"]: protocol["datasets"]
                ["candidate_pool"]["revision"],
            },
            model_revision=protocol["model"]["revision"],
            extra={"gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__},
        )
        _write_json_exclusive(run_dir / "resolved_recipe.json", recipe)
        return run_dir, manifest

    run_dir = resume_run_dir.resolve()
    run_dir.relative_to(output_root)
    manifest = _read_json(run_dir / "manifest.json")
    if manifest["config"] != run_config or manifest["seed"] != seed:
        raise ValueError("resume run differs from the frozen calibration contract")
    if manifest["git_commit"] != _git_commit():
        raise ValueError("resume must use the calibration run's original commit")
    if _read_json(run_dir / "resolved_recipe.json") != recipe:
        raise ValueError("resolved calibration recipe changed")
    return run_dir, manifest


def _save_final_artifacts(
    *,
    run_dir: Path,
    protocol: dict[str, Any],
    config: dict[str, Any],
    profile: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    fixed_batch: dict[str, torch.Tensor],
    token_audit: list[dict[str, Any]],
    parameter_report: Any,
    optimizer_steps_planned: int,
    optimizer_steps: int,
    supervised_tokens_seen: int,
    response_loss_sum_seen: float,
    compute_seconds: float,
    wall_seconds: float,
    max_peak_memory: int,
    temperature_sample_count: int,
) -> dict[str, Any]:
    pre_reload_loss = mean_response_token_loss(model=model, batch=fixed_batch)
    final_dir = run_dir / "training_complete"
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
        protocol["model"]["repo_id"],
        revision=protocol["model"]["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=str(config["attention_implementation"]),
    )
    reloaded = PeftModel.from_pretrained(
        reloaded_base,
        adapter_dir,
        is_trainable=False,
    ).to("cuda")
    post_reload_loss = mean_response_token_loss(model=reloaded, batch=fixed_batch)
    reload_difference = abs(post_reload_loss - pre_reload_loss)
    expected_samples = expected_temperature_sample_count(
        optimizer_steps_planned=optimizer_steps_planned
    )
    metrics = {
        "status": "PASS" if reload_difference <= 1e-6 else "FAIL",
        "study_role": "engineering_calibration_only_excluded_from_formal_matrix",
        "profile": profile.name,
        "training_example_count": len(token_audit),
        "micro_batch_size": profile.micro_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "nominal_effective_batch_size": profile.nominal_effective_batch_size,
        "optimizer_steps_planned": optimizer_steps_planned,
        "optimizer_steps_completed": optimizer_steps,
        "supervised_tokens_seen": supervised_tokens_seen,
        "mean_response_token_loss_seen": response_loss_sum_seen / supervised_tokens_seen,
        "compute_seconds_excluding_monitor_and_checkpoint_io": compute_seconds,
        "wall_training_loop_seconds": wall_seconds,
        "supervised_tokens_per_compute_second": supervised_tokens_seen / compute_seconds,
        "supervised_tokens_per_wall_second": supervised_tokens_seen / wall_seconds,
        "temperature_sample_count": temperature_sample_count,
        "expected_uninterrupted_temperature_sample_count": expected_samples,
        "temperature_sampling_rule": "once_at_start_and_once_per_optimizer_boundary",
        "peak_training_memory_bytes": max_peak_memory,
        "peak_training_memory_gib": max_peak_memory / 1024**3,
        "trainable_parameters": parameter_report.trainable_parameters,
        "total_parameters": parameter_report.total_parameters,
        "adapter_model_sha256": adapter_sha256,
        "pre_reload_fixed_batch_loss": pre_reload_loss,
        "post_reload_fixed_batch_loss": post_reload_loss,
        "adapter_reload_loss_absolute_difference": reload_difference,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": "Calibration throughput and integrity only; no selector result.",
    }
    _write_json_exclusive(attempt / "calibration_metrics.json", metrics)
    _write_json_exclusive(attempt / "token_audit.json", token_audit)
    attempt.rename(final_dir)
    del reloaded, reloaded_base
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("configs/b500_cloud_v2_calibration_v1.json"),
    )
    parser.add_argument(
        "--profile",
        choices=["mb1_ga16", "mb2_ga8", "mb4_ga4", "mb8_ga2"],
        required=True,
    )
    parser.add_argument("--resume-run-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("cloud-v2 training calibration requires a BF16 CUDA GPU")
    _require_clean_git_worktree()
    calibration_path = args.calibration_config.resolve()
    config = read_json_object(calibration_path)
    profiles = validate_calibration_config(config)
    profile = profiles[args.profile]
    protocol_path = resolve_frozen_artifact(
        repo_root=ROOT,
        binding=config["protocol_config"],
        label="protocol config",
    )
    recipe_path = resolve_frozen_artifact(
        repo_root=ROOT,
        binding=config["base_recipe_config"],
        label="base recipe config",
    )
    selection_path = resolve_frozen_artifact(
        repo_root=ROOT,
        binding=config["random_selection_manifest"],
        label="random selection manifest",
    )
    data_manifest_dir = validate_data_bindings(repo_root=ROOT, config=config)
    protocol = _read_json(protocol_path)
    base_recipe = _read_json(recipe_path)
    selection_manifest = _read_json(selection_path)
    selected_all = validate_selection_manifest(
        selection_manifest,
        expected_strategy="random",
        expected_budget=int(base_recipe["selection"]["budget"]),
        expected_selection_seed=int(base_recipe["selection"]["selection_seed"]),
    )
    selected = selected_all[: int(config["training_example_count"])]
    if len(selected) != 64:
        raise ValueError("frozen random manifest cannot supply 64 calibration examples")
    recipe = copy.deepcopy(base_recipe)
    recipe["protocol_version"] = "b500-cloud-v2-training-calibration-fixed-v1"
    recipe["training"]["epochs"] = int(config["training_epochs"])
    recipe["training"]["micro_batch_size"] = profile.micro_batch_size
    recipe["training"]["gradient_accumulation_steps"] = (
        profile.gradient_accumulation_steps
    )
    recipe["training"]["nominal_effective_batch_size"] = (
        profile.nominal_effective_batch_size
    )
    recipe["training"]["loss_normalization"] = config["loss_normalization"]
    recipe["training"]["attention_implementation"] = config[
        "attention_implementation"
    ]
    recipe["training"]["gradient_checkpointing"] = config[
        "gradient_checkpointing"
    ]
    run_config = {
        **calibration_run_config(
            payload=config,
            profile=profile,
            generation_batch_size=None,
        ),
        "entry_point": "scripts/run_b500_cloud_v2_train_calibration_fixed.py",
        "calibration_config_file_sha256": file_sha256(calibration_path),
        "protocol_config_sha256": file_sha256(protocol_path),
        "base_recipe_config_sha256": file_sha256(recipe_path),
        "selection_manifest_sha256": file_sha256(selection_path),
        "selected_id_sha256": selected_id_sha256(selected),
        "resolved_training": recipe["training"],
    }
    output_root = repository_path(
        ROOT,
        str(config["training_output_root"]),
        label="training calibration output root",
    )
    seed = int(config["training_seed"])
    run_dir, manifest = _create_or_resume_run(
        output_root=output_root,
        run_config=run_config,
        recipe=recipe,
        seed=seed,
        profile_name=profile.name,
        protocol=protocol,
        resume_run_dir=args.resume_run_dir,
    )
    completed_metrics = run_dir / "training_complete" / "calibration_metrics.json"
    if completed_metrics.is_file():
        print(completed_metrics.read_text(encoding="utf-8"))
        return

    guards = config["resource_guards"]
    start_gpu = _gpu_sample()
    temperature_sample_count = 1
    if start_gpu["temperature_c"] > float(guards["start_max_temperature_c"]):
        raise RuntimeError("GPU is above the frozen calibration start temperature")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model"]["repo_id"],
        revision=protocol["model"]["revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples, token_audit, _, _ = _prepare_training_data(
        protocol=protocol,
        recipe=recipe,
        selected=selected,
        data_manifest_dir=data_manifest_dir,
        tokenizer=tokenizer,
    )
    collator = ResponseOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_calibration_model(
        protocol=protocol,
        training=recipe["training"],
        attention_implementation=str(config["attention_implementation"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        device=device,
    )
    parameter_report = audit_lora_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(recipe["training"]["learning_rate"]),
        weight_decay=float(recipe["training"]["weight_decay"]),
    )
    orders = deterministic_epoch_orders(
        example_count=len(train_examples),
        epochs=int(recipe["training"]["epochs"]),
        seed=seed,
    )
    micro_batches = build_training_micro_batches(
        epoch_orders=orders,
        micro_batch_size=profile.micro_batch_size,
    )
    optimizer_steps_planned = optimizer_steps_for_examples(
        example_count=len(train_examples) * int(recipe["training"]["epochs"]),
        nominal_effective_batch_size=profile.nominal_effective_batch_size,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(
            optimizer_steps_planned * float(recipe["training"]["warmup_ratio"])
        ),
        num_training_steps=optimizer_steps_planned,
    )
    binding = {
        "run_config_hash": manifest["config_hash"],
        "git_commit": manifest["git_commit"],
        "calibration_profile": profile.name,
        "seed": seed,
        "selected_id_sha256": selected_id_sha256(selected),
    }
    checkpoint_dir = run_dir / "checkpoints"
    latest = load_latest_checkpoint(
        checkpoint_directory=checkpoint_dir,
        expected_binding=binding,
    )
    if latest is None:
        state = calibration_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng_state=_rng_state(),
            next_micro_batch_index=0,
            optimizer_steps=0,
            supervised_tokens_seen=0,
            response_loss_sum_seen=0.0,
            compute_seconds_completed=0.0,
            wall_seconds_completed=0.0,
            max_peak_memory_bytes_seen=int(torch.cuda.max_memory_allocated()),
            temperature_sample_count=temperature_sample_count,
        )
        saved = write_immutable_checkpoint(
            checkpoint_directory=checkpoint_dir,
            state=state,
            binding=binding,
        )
        _append_jsonl(
            run_dir / "runtime_events.jsonl",
            {"event": "initial_checkpoint", **saved["sidecar"]},
        )
    else:
        state, sidecar = latest
        set_peft_model_state_dict(model, state["adapter_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state"])
        _restore_rng_state(state["rng_state"])
        if int(state.get("pending_micro_batches", -1)) != 0:
            raise ValueError("calibration checkpoint is not at an optimizer boundary")
        temperature_sample_count += int(state.get("temperature_sample_count", 0))
        _append_jsonl(
            run_dir / "runtime_events.jsonl",
            {"event": "checkpoint_resumed", **sidecar},
        )

    next_micro_batch_index = int(state["next_micro_batch_index"])
    optimizer_steps = int(state["optimizer_steps"])
    supervised_tokens_seen = int(state["supervised_tokens_seen"])
    response_loss_sum_seen = float(state["response_loss_sum_seen"])
    compute_seconds = float(state["compute_seconds_completed"])
    previous_wall_seconds = float(state["wall_seconds_completed"])
    max_peak_memory = int(state["max_peak_memory_bytes_seen"])
    pending_micro_batches = 0
    pending_response_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    gradient_audited = optimizer_steps > 0
    wall_started = time.perf_counter()
    compute_window_started = time.perf_counter()
    model.train()
    for micro_batch_index in range(next_micro_batch_index, len(micro_batches)):
        plan = micro_batches[micro_batch_index]
        batch = collator([train_examples[item.example_index] for item in plan])
        batch = _to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss_sums, token_counts = shifted_response_loss_sums(
            logits=outputs.logits,
            labels=batch["labels"],
        )
        loss_sum = loss_sums.sum()
        token_count = int(token_counts.sum().item())
        loss_sum.backward()
        pending_micro_batches += 1
        pending_response_tokens += token_count
        supervised_tokens_seen += token_count
        response_loss_sum_seen += float(loss_sum.detach().item())
        if not gradient_audited:
            audit_lora_gradients(model)
            gradient_audited = True

        final_micro_batch = micro_batch_index + 1 == len(micro_batches)
        if pending_micro_batches == profile.gradient_accumulation_steps or final_micro_batch:
            normalize_gradients_by_token_count(
                trainable_parameters,
                response_token_count=pending_response_tokens,
            )
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                float(recipe["training"]["gradient_clipping"]),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending_micro_batches = 0
            pending_response_tokens = 0
            torch.cuda.synchronize()
            compute_seconds += time.perf_counter() - compute_window_started
            max_peak_memory = max(max_peak_memory, int(torch.cuda.max_memory_allocated()))
            sample = _gpu_sample()
            temperature_sample_count += 1
            wall_seconds = previous_wall_seconds + time.perf_counter() - wall_started
            state = calibration_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rng_state=_rng_state(),
                next_micro_batch_index=micro_batch_index + 1,
                optimizer_steps=optimizer_steps,
                supervised_tokens_seen=supervised_tokens_seen,
                response_loss_sum_seen=response_loss_sum_seen,
                compute_seconds_completed=compute_seconds,
                wall_seconds_completed=wall_seconds,
                max_peak_memory_bytes_seen=max_peak_memory,
                temperature_sample_count=temperature_sample_count,
            )
            checkpoint_due = should_write_checkpoint(
                optimizer_step=optimizer_steps,
                optimizer_steps_planned=optimizer_steps_planned,
                checkpoint_every_optimizer_steps=int(
                    config["checkpoint_every_optimizer_steps"]
                ),
            )
            hard_stop = sample["temperature_c"] >= float(
                guards["hard_stop_temperature_c"]
            )
            if checkpoint_due or hard_stop:
                saved = write_immutable_checkpoint(
                    checkpoint_directory=checkpoint_dir,
                    state=state,
                    binding=binding,
                )
                _append_jsonl(
                    run_dir / "runtime_events.jsonl",
                    {
                        "event": "emergency_checkpoint" if hard_stop else "checkpoint_saved",
                        **saved["sidecar"],
                    },
                )
            if hard_stop:
                raise RuntimeError("cloud-v2 calibration reached its hard temperature stop")
            compute_window_started = time.perf_counter()

        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        if peak_gib > float(guards["max_peak_gpu_memory_gib"]):
            raise RuntimeError("cloud-v2 calibration exceeded its peak GPU memory guard")

    if pending_micro_batches or pending_response_tokens:
        raise AssertionError("final effective-batch remainder was not committed")
    if optimizer_steps != optimizer_steps_planned:
        raise ValueError("optimizer step count differs from the frozen calibration plan")
    wall_seconds = previous_wall_seconds + time.perf_counter() - wall_started
    fixed_batch = _to_device(
        collator([train_examples[item.example_index] for item in micro_batches[0]]),
        device,
    )
    metrics = _save_final_artifacts(
        run_dir=run_dir,
        protocol=protocol,
        config=config,
        profile=profile,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        fixed_batch=fixed_batch,
        token_audit=token_audit,
        parameter_report=parameter_report,
        optimizer_steps_planned=optimizer_steps_planned,
        optimizer_steps=optimizer_steps,
        supervised_tokens_seen=supervised_tokens_seen,
        response_loss_sum_seen=response_loss_sum_seen,
        compute_seconds=compute_seconds,
        wall_seconds=wall_seconds,
        max_peak_memory=max_peak_memory,
        temperature_sample_count=temperature_sample_count,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
