"""Run exactly one sealed cloud-v2 B=500 method/seed cell."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
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
    _free_disk_gib,
    _free_system_memory_gib,
    _git_commit,
    _global_job_lock,
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
from run_cloud_v2_generation_worker import _gpu_uuid  # noqa: E402

from eg_sft.evaluation.cloud_v2_batching import append_jsonl_rows_fsynced  # noqa: E402
from eg_sft.evaluation.formal_two_worker import (  # noqa: E402
    formal_shards,
    merge_formal_worker_outputs,
)
from eg_sft.evaluation.resumable import validate_completed_prefix  # noqa: E402
from eg_sft.experiment.cloud_v2_formal import (  # noqa: E402
    FORMAL_METHODS,
    FORMAL_SEEDS,
    engineering_stdout_payload,
    resolve_formal_contract,
)
from eg_sft.experiment.cloud_v2_train_runtime import (  # noqa: E402
    build_calibration_model,
    calibration_checkpoint_payload,
    mean_response_token_loss,
)
from eg_sft.experiment.formal_runtime import (  # noqa: E402
    deterministic_epoch_orders,
    load_latest_checkpoint,
    write_immutable_checkpoint,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402
from eg_sft.training.effective_batch import (  # noqa: E402
    build_training_micro_batches,
    normalize_gradients_by_token_count,
    optimizer_steps_for_examples,
    shifted_response_loss_sums,
    should_write_checkpoint,
)
from eg_sft.training.lora_audit import audit_lora_gradients, audit_lora_parameters  # noqa: E402
from eg_sft.training.response_only import ResponseOnlyCollator  # noqa: E402


def _resource_preflight(contract: dict[str, Any]) -> dict[str, Any]:
    resources = contract["config"]["resources"]
    gpu = _gpu_sample()
    memory = _free_system_memory_gib()
    disk = _free_disk_gib(ROOT)
    if gpu["memory_used_mib"] >= 512:
        raise RuntimeError("formal cell requires an otherwise idle GPU")
    if memory < float(resources["min_free_system_memory_gib"]):
        raise RuntimeError("insufficient free system memory")
    if disk < float(resources["min_free_disk_gib"]):
        raise RuntimeError("insufficient free disk for immutable formal artifacts")
    if gpu["temperature_c"] >= float(resources["hard_stop_temperature_c"]):
        raise RuntimeError("GPU is already at the formal hard-stop temperature")
    return {
        "gpu": gpu,
        "gpu_uuid": _gpu_uuid(int(contract["config"]["evaluation"]["cuda_device_index"])),
        "free_system_memory_gib": memory,
        "free_disk_gib": disk,
    }


def _resolved_recipe(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = copy.deepcopy(contract["base_recipe"])
    frozen = contract["config"]["training"]
    training = recipe["training"]
    for field in ("epochs", "max_length", "micro_batch_size", "gradient_accumulation_steps"):
        training[field] = frozen[field]
    training["nominal_effective_batch_size"] = frozen["nominal_effective_batch_size"]
    training["loss_normalization"] = frozen["loss_normalization"]
    training["attention_implementation"] = frozen["attention_implementation"]
    training["gradient_checkpointing"] = frozen["gradient_checkpointing"]
    recipe["protocol_version"] = "cloud-v2-formal-b500-training-v1"
    return recipe


def _create_or_resume_run(
    *,
    contract: dict[str, Any],
    recipe: dict[str, Any],
    resume_run_dir: Path | None,
    command: list[str],
    resources: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    selection = contract["selection"]
    run_config = {
        "formal_protocol_version": contract["config"]["formal_protocol_version"],
        "formal_config_sha256": contract["config_sha256"],
        "method": contract["method"],
        "selection_manifest_sha256": selection["file_sha256"],
        "selected_id_sha256": selection["selected_id_sha256"],
        "protocol_config_sha256": file_sha256(contract["protocol_path"]),
        "base_recipe_config_sha256": file_sha256(contract["base_recipe_path"]),
        "training": recipe["training"],
        "evaluation": contract["config"]["evaluation"],
        "sealed_result_policy": {
            "stdout_accuracy": False,
            "stdout_method_comparison": False,
            "sealed_analysis_after_nine_audits": True,
        },
    }
    if resume_run_dir is None:
        if contract["output_root"].is_dir():
            for manifest_path in contract["output_root"].glob("*/manifest.json"):
                existing = _read_json(manifest_path)
                if (
                    existing.get("seed") == contract["seed"]
                    and existing.get("config", {}).get("method") == contract["method"]
                ):
                    raise FileExistsError(
                        "formal cell already exists; use --resume-run-dir explicitly"
                    )
        run_dir, manifest = create_run_manifest(
            output_root=contract["output_root"],
            repo_root=ROOT,
            stage=f"cloud_v2_formal_{contract['method']}",
            config=run_config,
            seed=contract["seed"],
            command=command,
            dataset_revisions={
                contract["protocol"]["datasets"]["gsm8k"]["repo_id"]: contract[
                    "protocol"
                ]["datasets"]["gsm8k"]["revision"],
                contract["protocol"]["datasets"]["candidate_pool"]["repo_id"]: contract[
                    "protocol"
                ]["datasets"]["candidate_pool"]["revision"],
            },
            model_revision=contract["protocol"]["model"]["revision"],
            extra={
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_uuid": resources["gpu_uuid"],
                "torch": torch.__version__,
            },
        )
        _write_json_exclusive(run_dir / "resolved_recipe.json", recipe)
        return run_dir, manifest
    run_dir = resume_run_dir.resolve()
    run_dir.relative_to(contract["output_root"])
    manifest = _read_json(run_dir / "manifest.json")
    if manifest["config"] != run_config or manifest["seed"] != contract["seed"]:
        raise ValueError("formal resume contract changed")
    if manifest["git_commit"] != _git_commit():
        raise ValueError("formal resume requires its original git commit")
    if _read_json(run_dir / "resolved_recipe.json") != recipe:
        raise ValueError("resolved formal recipe changed")
    return run_dir, manifest


def _save_training_complete(
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
        raise RuntimeError("formal adapter reload loss check failed")
    metrics = {
        "status": "PASS",
        "selected_count": len(token_audit),
        "epochs": int(recipe["training"]["epochs"]),
        "micro_batch_size": int(recipe["training"]["micro_batch_size"]),
        "gradient_accumulation_steps": int(
            recipe["training"]["gradient_accumulation_steps"]
        ),
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
    attempt.rename(final_dir)
    del reloaded, reloaded_base
    gc.collect()
    torch.cuda.empty_cache()
    return final_dir


def _train(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    recipe: dict[str, Any],
) -> Path:
    final_dir = run_dir / "training_complete"
    if final_dir.is_dir():
        return final_dir
    seed = int(contract["seed"])
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        contract["protocol"]["model"]["repo_id"],
        revision=contract["protocol"]["model"]["revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples, token_audit, development_loader, _ = _prepare_training_data(
        protocol=contract["protocol"],
        recipe=recipe,
        selected=contract["selection"]["selected"],
        data_manifest_dir=contract["data_dir"],
        tokenizer=tokenizer,
    )
    device = torch.device("cuda")
    collator = ResponseOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_calibration_model(
        protocol=contract["protocol"],
        training=recipe["training"],
        attention_implementation=str(recipe["training"]["attention_implementation"]),
        gradient_checkpointing=bool(recipe["training"]["gradient_checkpointing"]),
        device=device,
    )
    parameter_report = audit_lora_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(recipe["training"]["learning_rate"]),
        weight_decay=float(recipe["training"]["weight_decay"]),
    )
    epochs = int(recipe["training"]["epochs"])
    orders = deterministic_epoch_orders(
        example_count=len(train_examples), epochs=epochs, seed=seed
    )
    micro_batches = build_training_micro_batches(
        epoch_orders=orders,
        micro_batch_size=int(recipe["training"]["micro_batch_size"]),
    )
    optimizer_steps_planned = optimizer_steps_for_examples(
        example_count=len(train_examples) * epochs,
        nominal_effective_batch_size=int(
            recipe["training"]["nominal_effective_batch_size"]
        ),
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
        "method": contract["method"],
        "seed": seed,
        "selected_id_sha256": contract["selection"]["selected_id_sha256"],
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
            temperature_sample_count=0,
        )
        write_immutable_checkpoint(
            checkpoint_directory=checkpoint_dir, state=state, binding=binding
        )
    else:
        state, _ = latest
        set_peft_model_state_dict(model, state["adapter_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state"])
        _restore_rng_state(state["rng_state"])
        if int(state.get("pending_micro_batches", -1)) != 0:
            raise ValueError("formal checkpoint is not at an optimizer boundary")
    next_micro_batch = int(state["next_micro_batch_index"])
    optimizer_steps = int(state["optimizer_steps"])
    supervised_tokens_seen = int(state["supervised_tokens_seen"])
    response_loss_sum_seen = float(state["response_loss_sum_seen"])
    prior_wall_seconds = float(state["wall_seconds_completed"])
    max_peak_memory = int(state["max_peak_memory_bytes_seen"])
    pending_micro_batches = 0
    pending_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    gradient_audited = optimizer_steps > 0
    started = time.perf_counter()
    model.train()
    for micro_index in range(next_micro_batch, len(micro_batches)):
        plan = micro_batches[micro_index]
        batch = _to_device(
            collator([train_examples[item.example_index] for item in plan]), device
        )
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss_sums, token_counts = shifted_response_loss_sums(
            logits=outputs.logits, labels=batch["labels"]
        )
        loss_sum = loss_sums.sum()
        token_count = int(token_counts.sum().item())
        loss_sum.backward()
        pending_micro_batches += 1
        pending_tokens += token_count
        supervised_tokens_seen += token_count
        response_loss_sum_seen += float(loss_sum.detach().item())
        if not gradient_audited:
            audit_lora_gradients(model)
            gradient_audited = True
        final_micro = micro_index + 1 == len(micro_batches)
        if (
            pending_micro_batches
            == int(recipe["training"]["gradient_accumulation_steps"])
            or final_micro
        ):
            normalize_gradients_by_token_count(
                trainable_parameters, response_token_count=pending_tokens
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
            pending_tokens = 0
            max_peak_memory = max(max_peak_memory, int(torch.cuda.max_memory_allocated()))
            wall_seconds = prior_wall_seconds + time.perf_counter() - started
            state = calibration_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rng_state=_rng_state(),
                next_micro_batch_index=micro_index + 1,
                optimizer_steps=optimizer_steps,
                supervised_tokens_seen=supervised_tokens_seen,
                response_loss_sum_seen=response_loss_sum_seen,
                compute_seconds_completed=wall_seconds,
                wall_seconds_completed=wall_seconds,
                max_peak_memory_bytes_seen=max_peak_memory,
                temperature_sample_count=optimizer_steps,
            )
            if should_write_checkpoint(
                optimizer_step=optimizer_steps,
                optimizer_steps_planned=optimizer_steps_planned,
                checkpoint_every_optimizer_steps=int(
                    contract["config"]["training"]["checkpoint_every_optimizer_steps"]
                ),
            ):
                saved = write_immutable_checkpoint(
                    checkpoint_directory=checkpoint_dir,
                    state=state,
                    binding=binding,
                )
                _append_jsonl(
                    run_dir / "runtime_events.jsonl",
                    {"event": "checkpoint_saved", **saved["sidecar"]},
                )
            gpu = _gpu_sample()
            if gpu["temperature_c"] >= float(
                contract["config"]["resources"]["hard_stop_temperature_c"]
            ):
                raise RuntimeError("formal training reached the hard temperature stop")
            print(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "stage": "formal_training",
                        "optimizer_steps": f"{optimizer_steps}/{optimizer_steps_planned}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if torch.cuda.max_memory_allocated() / 1024**3 > float(
            contract["config"]["resources"]["max_training_peak_allocated_gib"]
        ):
            raise RuntimeError("formal training exceeded its memory guard")
    if pending_micro_batches or pending_tokens or optimizer_steps != optimizer_steps_planned:
        raise ValueError("formal training did not finish on its frozen optimizer boundary")
    fixed_batch = _to_device(next(iter(development_loader)), device)
    return _save_training_complete(
        run_dir=run_dir,
        contract=contract,
        recipe=recipe,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        fixed_batch=fixed_batch,
        parameter_report=parameter_report,
        token_audit=token_audit,
        state=state,
    )


def _evaluate(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    training_dir: Path,
) -> Path:
    merged_dir = run_dir / "evaluation" / "merged"
    final_metrics = merged_dir / "metrics.json"
    if final_metrics.is_file():
        return merged_dir
    shards = formal_shards(contract["config"]["evaluation"])
    adapter_sha256 = file_sha256(training_dir / "adapter" / "adapter_model.safetensors")
    processes: list[tuple[str, subprocess.Popen, Any]] = []
    completed_workers: list[str] = []
    for shard in shards:
        worker_dir = run_dir / "evaluation" / "workers" / shard.shard_id
        if (worker_dir / "metrics.json").is_file():
            completed_workers.append(shard.shard_id)
            continue
        log_path = run_dir / "evaluation" / f"{shard.shard_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open(
            "a" if log_path.exists() else "x", encoding="utf-8", newline="\n"
        )
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_cloud_v2_formal_eval_worker.py"),
            "--run-dir",
            str(run_dir),
            "--config",
            str(contract["config_path"]),
            "--shard-id",
            shard.shard_id,
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard.shard_id, process, log_handle))
    return_codes = {shard_id: 0 for shard_id in completed_workers}
    for shard_id, process, log_handle in processes:
        return_codes[shard_id] = process.wait()
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()
    failures = {key: value for key, value in return_codes.items() if value != 0}
    if failures:
        _append_jsonl(
            run_dir / "evaluation" / "launcher_failures.jsonl",
            {
                "event": "formal_eval_worker_failure",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "return_codes": return_codes,
                "pass_merge_created": False,
            },
        )
        raise RuntimeError(f"formal evaluation worker failure prevented merge: {failures}")
    all_records = read_jsonl(contract["data_dir"] / "gsm8k_records.jsonl")
    frozen_records = sorted(
        (
            row
            for row in all_records
            if row["protocol_split"] == contract["config"]["evaluation"]["split"]
        ),
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    worker_payloads = {
        shard.shard_id: {
            "manifest": _read_json(
                run_dir / "evaluation" / "workers" / shard.shard_id / "manifest.json"
            ),
            "metrics": _read_json(
                run_dir / "evaluation" / "workers" / shard.shard_id / "metrics.json"
            ),
            "rows": read_jsonl(
                run_dir / "evaluation" / "workers" / shard.shard_id / "raw_outputs.jsonl"
            ),
        }
        for shard in shards
    }
    merged_rows, report = merge_formal_worker_outputs(
        frozen_records=frozen_records,
        shards=shards,
        worker_payloads=worker_payloads,
    )
    if report["adapter_model_sha256"] != adapter_sha256:
        raise ValueError("merged formal evaluation adapter hash changed")
    merged_dir.mkdir(parents=True, exist_ok=True)
    raw_path = merged_dir / "raw_outputs.jsonl"
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    next_index = validate_completed_prefix(
        completed_rows=existing, frozen_records=frozen_records
    )
    append_jsonl_rows_fsynced(raw_path, merged_rows[next_index:])
    completed = read_jsonl(raw_path)
    validate_completed_prefix(completed_rows=completed, frozen_records=frozen_records)
    if len(completed) != 1319:
        raise ValueError("formal merged output is not exactly 1319 rows")
    metrics = {
        **report,
        "source_run_id": manifest["run_id"],
        "raw_outputs_sha256": file_sha256(raw_path),
        "accuracy_withheld": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json_exclusive(merged_dir / "metrics.json", metrics)
    _write_json_exclusive(
        merged_dir / "manifest.json",
        {
            "status": "PASS",
            "source_run_id": manifest["run_id"],
            "formal_config_sha256": contract["config_sha256"],
            "adapter_model_sha256": adapter_sha256,
            "gpu_uuid": report["gpu_uuid"],
            "raw_outputs_sha256": file_sha256(raw_path),
            "accuracy_withheld": True,
        },
    )
    return merged_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud_v2_formal_b500_single_cell_v1.json"),
    )
    parser.add_argument("--method", choices=FORMAL_METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal cloud-v2 cell requires a BF16 CUDA GPU")
    _require_clean_git_worktree()
    contract = resolve_formal_contract(
        repo_root=ROOT,
        config_path=args.config.resolve(),
        method=args.method,
        seed=args.seed,
    )
    resources = _resource_preflight(contract)
    hashes = {
        "formal_config_sha256": contract["config_sha256"],
        "selection_manifest_sha256": contract["selection"]["file_sha256"],
        "selected_id_sha256": contract["selection"]["selected_id_sha256"],
    }
    if args.preflight_only:
        print(
            json.dumps(
                engineering_stdout_payload(
                    status="READY", run_id=None, hashes=hashes, stage="preflight"
                ),
                sort_keys=True,
            )
        )
        return
    recipe = _resolved_recipe(contract)
    with _global_job_lock(contract["output_root"]):
        run_dir, manifest = _create_or_resume_run(
            contract=contract,
            recipe=recipe,
            resume_run_dir=args.resume_run_dir,
            command=[sys.executable, *sys.argv],
            resources=resources,
        )
        _append_jsonl(
            run_dir / "invocations.jsonl",
            {
                "event": "formal_cell_invocation_start",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "git_commit": _git_commit(),
                "resource_preflight": resources,
            },
        )
        training_dir = _train(
            run_dir=run_dir,
            manifest=manifest,
            contract=contract,
            recipe=recipe,
        )
        evaluation_dir = _evaluate(
            run_dir=run_dir,
            manifest=manifest,
            contract=contract,
            training_dir=training_dir,
        )
        completion_path = run_dir / "cell_complete.json"
        if not completion_path.exists():
            completion = {
                "status": "PASS",
                "run_id": manifest["run_id"],
                "training_metrics_sha256": file_sha256(
                    training_dir / "training_metrics.json"
                ),
                "adapter_model_sha256": file_sha256(
                    training_dir / "adapter" / "adapter_model.safetensors"
                ),
                "evaluation_metrics_sha256": file_sha256(
                    evaluation_dir / "metrics.json"
                ),
                "raw_outputs_sha256": file_sha256(
                    evaluation_dir / "raw_outputs.jsonl"
                ),
                "record_count": 1319,
                "accuracy_withheld": True,
                "next_cell_started": False,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            _write_json_exclusive(completion_path, completion)
        completion = _read_json(completion_path)
        print(
            json.dumps(
                engineering_stdout_payload(
                    status="COMPLETE",
                    run_id=manifest["run_id"],
                    hashes={
                        "adapter_model_sha256": completion["adapter_model_sha256"],
                        "raw_outputs_sha256": completion["raw_outputs_sha256"],
                        "cell_complete_sha256": file_sha256(completion_path),
                    },
                    stage="formal_cell",
                ),
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
