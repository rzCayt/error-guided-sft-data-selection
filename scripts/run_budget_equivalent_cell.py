"""Run exactly one sealed budget-equivalent Phase 1 cell."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import set_peft_model_state_dict
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import (  # noqa: E402
    _append_jsonl,
    _git_commit,
    _global_job_lock,
    _optimizer_to_device,
    _prepare_training_data,
    _read_json,
    _require_clean_git_worktree,
    _restore_rng_state,
    _rng_state,
    _to_device,
    _write_json_exclusive,
)
from run_cloud_v2_formal_cell import (  # noqa: E402
    _resource_preflight,
    _save_training_complete,
)

from eg_sft.evaluation.cloud_v2_batching import append_jsonl_rows_fsynced  # noqa: E402
from eg_sft.evaluation.formal_two_worker import (  # noqa: E402
    formal_shards,
    merge_formal_worker_outputs,
)
from eg_sft.evaluation.resumable import validate_completed_prefix  # noqa: E402
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    resolve_phase1_contract,
)
from eg_sft.experiment.cloud_v2_train_runtime import (  # noqa: E402
    build_calibration_model,
    calibration_checkpoint_payload,
)
from eg_sft.experiment.formal_runtime import (  # noqa: E402
    deterministic_epoch_orders,
    load_latest_checkpoint,
    write_immutable_checkpoint,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402
from eg_sft.training.effective_batch import (  # noqa: E402
    normalize_gradients_by_token_count,
    shifted_response_loss_sums,
    should_write_checkpoint,
)
from eg_sft.training.lora_audit import audit_lora_gradients, audit_lora_parameters  # noqa: E402
from eg_sft.training.response_only import ResponseOnlyCollator  # noqa: E402
from eg_sft.training.token_budget import (  # noqa: E402
    balanced_optimizer_step_plan,
    micro_batches_for_step,
    optimizer_step_token_audit,
)


def _resolved_recipe(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = copy.deepcopy(contract["base_recipe"])
    frozen = contract["config"]["training"]
    training = recipe["training"]
    for field in (
        "epochs",
        "optimizer_steps",
        "max_length",
        "micro_batch_size",
        "loss_normalization",
        "attention_implementation",
        "gradient_checkpointing",
    ):
        training[field] = frozen[field]
    recipe["protocol_version"] = "budget-equivalent-phase1-training-v3"
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
        "phase1_protocol_version": contract["config"]["phase1_protocol_version"],
        "phase1_config_sha256": contract["config_sha256"],
        "cell_id": contract["cell_id"],
        "replicate_index": contract["replicate_index"],
        "method": contract["method"],
        "selection_manifest_sha256": selection["file_sha256"],
        "selected_id_sha256": selection["selected_id_sha256"],
        "training": recipe["training"],
        "evaluation": contract["config"]["evaluation"],
        "sealed_result_policy": {
            "stdout_accuracy": False,
            "stdout_method_comparison": False,
            "unblind_after_audited_cells": 16,
        },
    }
    if resume_run_dir is None:
        if contract["output_root"].is_dir():
            for manifest_path in contract["output_root"].glob("*/manifest.json"):
                existing = _read_json(manifest_path)
                if existing.get("config", {}).get("cell_id") == contract["cell_id"]:
                    raise FileExistsError(
                        "Phase 1 cell already exists; use --resume-run-dir explicitly"
                    )
        run_dir, manifest = create_run_manifest(
            output_root=contract["output_root"],
            repo_root=ROOT,
            stage=f"budget_equivalent_{contract['cell_id']}",
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
        raise ValueError("Phase 1 resume contract changed")
    if manifest["git_commit"] != _git_commit():
        raise ValueError("Phase 1 resume requires its original git commit")
    if _read_json(run_dir / "resolved_recipe.json") != recipe:
        raise ValueError("resolved Phase 1 recipe changed")
    return run_dir, manifest


def _load_step_token_log(path: Path, *, expected_steps: int) -> list[int]:
    if not path.exists():
        if expected_steps:
            raise ValueError("checkpoint exists without optimizer-step token log")
        return []
    rows = read_jsonl(path)
    if len(rows) != expected_steps:
        raise ValueError("optimizer-step token log differs from checkpoint progress")
    for index, row in enumerate(rows, start=1):
        if int(row["optimizer_step"]) != index:
            raise ValueError("optimizer-step token log is not contiguous")
    return [int(row["response_supervision_tokens"]) for row in rows]


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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(recipe["training"]["learning_rate"]),
        weight_decay=float(recipe["training"]["weight_decay"]),
    )
    orders = deterministic_epoch_orders(
        example_count=len(train_examples),
        epochs=int(recipe["training"]["epochs"]),
        seed=seed,
    )
    planned_steps = int(recipe["training"]["optimizer_steps"])
    step_plan = balanced_optimizer_step_plan(
        epoch_orders=orders,
        optimizer_steps=planned_steps,
    )
    flat_micro_batches = []
    step_end_indices: set[int] = set()
    for step_items in step_plan:
        flat_micro_batches.extend(
            micro_batches_for_step(
                step_items,
                micro_batch_size=int(recipe["training"]["micro_batch_size"]),
            )
        )
        step_end_indices.add(len(flat_micro_batches))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(planned_steps * float(recipe["training"]["warmup_ratio"])),
        num_training_steps=planned_steps,
    )
    binding = {
        "run_config_hash": manifest["config_hash"],
        "git_commit": manifest["git_commit"],
        "cell_id": contract["cell_id"],
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
            raise ValueError("Phase 1 checkpoint is not at an optimizer boundary")
    next_micro_batch = int(state["next_micro_batch_index"])
    optimizer_steps = int(state["optimizer_steps"])
    step_token_path = run_dir / "optimizer_step_tokens.jsonl"
    step_token_counts = _load_step_token_log(step_token_path, expected_steps=optimizer_steps)
    supervised_tokens_seen = int(state["supervised_tokens_seen"])
    response_loss_sum_seen = float(state["response_loss_sum_seen"])
    prior_wall_seconds = float(state["wall_seconds_completed"])
    max_peak_memory = int(state["max_peak_memory_bytes_seen"])
    pending_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    gradient_audited = optimizer_steps > 0
    started = time.perf_counter()
    model.train()
    for micro_index in range(next_micro_batch, len(flat_micro_batches)):
        plan = flat_micro_batches[micro_index]
        batch = _to_device(
            collator([train_examples[item.example_index] for item in plan]), device
        )
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss_sums, token_counts = shifted_response_loss_sums(
            logits=outputs.logits, labels=batch["labels"]
        )
        loss_sum = loss_sums.sum()
        token_count = int(token_counts.sum().item())
        loss_sum.backward()
        pending_tokens += token_count
        supervised_tokens_seen += token_count
        response_loss_sum_seen += float(loss_sum.detach().item())
        if not gradient_audited:
            audit_lora_gradients(model)
            gradient_audited = True
        if micro_index + 1 in step_end_indices:
            normalize_gradients_by_token_count(
                trainable, response_token_count=pending_tokens
            )
            torch.nn.utils.clip_grad_norm_(
                trainable, float(recipe["training"]["gradient_clipping"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            step_token_counts.append(pending_tokens)
            _append_jsonl(
                step_token_path,
                {
                    "optimizer_step": optimizer_steps,
                    "response_supervision_tokens": pending_tokens,
                },
            )
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
                optimizer_steps_planned=planned_steps,
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
            print(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "stage": "budget_equivalent_training",
                        "optimizer_steps": f"{optimizer_steps}/{planned_steps}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if pending_tokens or optimizer_steps != planned_steps:
        raise ValueError("Phase 1 training did not end at its frozen optimizer boundary")
    expected_exposure = sum(int(row["supervised_tokens"]) for row in token_audit) * int(
        recipe["training"]["epochs"]
    )
    token_budget_audit = optimizer_step_token_audit(
        step_token_counts=step_token_counts,
        expected_optimizer_steps=planned_steps,
        expected_exposure_tokens=expected_exposure,
        tolerance_fraction=0.005,
    )
    if not token_budget_audit["exposure_gate_passed"]:
        raise ValueError("Phase 1 response-token exposure gate failed")
    fixed_batch = _to_device(next(iter(development_loader)), device)
    completed = _save_training_complete(
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
    audit_path = completed / "token_budget_audit.json"
    if not audit_path.exists():
        _write_json_exclusive(audit_path, token_budget_audit)
    return completed


def _evaluate(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    training_dir: Path,
) -> Path:
    merged_dir = run_dir / "evaluation" / "merged"
    if (merged_dir / "metrics.json").is_file():
        return merged_dir
    shards = formal_shards(contract["config"]["evaluation"])
    adapter_sha256 = file_sha256(training_dir / "adapter" / "adapter_model.safetensors")
    processes: list[tuple[str, subprocess.Popen, Any]] = []
    completed_workers = []
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
            str(ROOT / "scripts" / "run_budget_equivalent_eval_worker.py"),
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
        raise RuntimeError(f"Phase 1 evaluation worker failure: {failures}")
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
                run_dir
                / "evaluation"
                / "workers"
                / shard.shard_id
                / "raw_outputs.jsonl"
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
        raise ValueError("merged evaluation adapter hash changed")
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
        raise ValueError("Phase 1 merged output is not exactly 1319 rows")
    metrics = {
        **report,
        "source_run_id": manifest["run_id"],
        "raw_outputs_sha256": file_sha256(raw_path),
        "accuracy_withheld": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json_exclusive(merged_dir / "metrics.json", metrics)
    return merged_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    contract = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=args.config.resolve(),
        cell_id=args.cell_id,
    )
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "stage": "contract",
                    "cell_id": contract["cell_id"],
                    "hashes": {
                        "config": contract["config_sha256"],
                        "selection": contract["selection"]["file_sha256"],
                    },
                },
                sort_keys=True,
            )
        )
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 1 cell requires a BF16 CUDA GPU")
    _require_clean_git_worktree()
    resources = _resource_preflight(contract)
    if args.preflight_only:
        print(json.dumps({"status": "READY", "stage": "gpu_preflight"}, sort_keys=True))
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
        completion = run_dir / "cell_complete.json"
        if not completion.exists():
            _write_json_exclusive(
                completion,
                {
                    "status": "PASS",
                    "cell_id": contract["cell_id"],
                    "run_id": manifest["run_id"],
                    "adapter_model_sha256": file_sha256(
                        training_dir / "adapter" / "adapter_model.safetensors"
                    ),
                    "token_budget_audit_sha256": file_sha256(
                        training_dir / "token_budget_audit.json"
                    ),
                    "raw_outputs_sha256": file_sha256(
                        evaluation_dir / "raw_outputs.jsonl"
                    ),
                    "record_count": 1319,
                    "accuracy_withheld": True,
                    "next_cell_started": False,
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                },
            )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "stage": "budget_equivalent_cell",
                    "cell_id": contract["cell_id"],
                    "accuracy_withheld": True,
                    "cell_complete_sha256": file_sha256(completion),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
