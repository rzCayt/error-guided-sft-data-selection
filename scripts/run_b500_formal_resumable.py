"""Run exactly one frozen B=500 job with immutable checkpoints and thermal gates."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    PROMPT_VERSION,
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.evaluation.resumable import (  # noqa: E402
    aggregate_gsm8k_metrics,
    validate_completed_prefix,
)
from eg_sft.experiment.b500_matrix import preflight_b500_matrix  # noqa: E402
from eg_sft.experiment.formal_runtime import (  # noqa: E402
    ThermalPolicy,
    deterministic_epoch_orders,
    load_latest_checkpoint,
    validate_execution_policy,
    write_immutable_checkpoint,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.training.b500 import (  # noqa: E402
    file_sha256,
    read_jsonl,
    selected_id_sha256,
    tokenize_tulu_candidate,
    validate_selection_manifest,
)
from eg_sft.training.lora_audit import (  # noqa: E402
    audit_lora_gradients,
    audit_lora_parameters,
)
from eg_sft.training.overfit import build_tokenized_overfit_examples  # noqa: E402
from eg_sft.training.response_only import ResponseOnlyCollator  # noqa: E402


class ThermalStop(RuntimeError):
    """Expected stop after the hard temperature threshold is observed."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _require_clean_git_worktree() -> None:
    process = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if process.stdout.strip():
        raise RuntimeError("formal run requires a clean git worktree")


def _gpu_sample() -> dict[str, float]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [row.strip() for row in process.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU row, received {rows}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 5:
        raise RuntimeError(f"unexpected nvidia-smi row: {rows[0]}")
    return {
        "temperature_c": float(values[0]),
        "memory_used_mib": float(values[1]),
        "memory_total_mib": float(values[2]),
        "utilization_percent": float(values[3]),
        "power_w": float(values[4]),
    }


def _free_system_memory_gib() -> float:
    if os.name == "nt":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.length = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.available_physical) / 1024**3
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return float(page_size * available_pages) / 1024**3


def _free_disk_gib(path: Path) -> float:
    import shutil

    return float(shutil.disk_usage(path).free) / 1024**3


@contextmanager
def _global_job_lock(output_root: Path) -> Iterator[None]:
    """Hold one OS-level lock without deleting or replacing a lock file."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "formal_job.lock"
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    "another formal B=500 job already holds the global lock"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(
                    "another formal B=500 job already holds the global lock"
                ) from error
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class ThermalController:
    def __init__(
        self,
        *,
        policy: ThermalPolicy,
        event_path: Path,
    ) -> None:
        self.policy = policy
        self.event_path = event_path
        self.max_temperature_c = 0.0

    def _sample(self) -> dict[str, float]:
        sample = _gpu_sample()
        self.max_temperature_c = max(
            self.max_temperature_c,
            float(sample["temperature_c"]),
        )
        return sample

    def wait_until_start_safe(self) -> None:
        sample = self._sample()
        if sample["memory_used_mib"] >= 512:
            raise RuntimeError("GPU memory became busy before model loading")
        if sample["temperature_c"] >= self.policy.hard_stop_at_c:
            raise ThermalStop(f"GPU is already at {sample['temperature_c']:.0f}C before work")
        while sample["temperature_c"] > self.policy.start_max_c:
            print(
                "start_cooldown "
                f"temperature_c={sample['temperature_c']:.0f} "
                f"start_max_c={self.policy.start_max_c}",
                flush=True,
            )
            time.sleep(self.policy.poll_seconds)
            sample = self._sample()

    def guard(self, *, stage: str, progress: int) -> dict[str, float]:
        sample = self._sample()
        if sample["temperature_c"] >= self.policy.hard_stop_at_c:
            event = {
                "event": "hard_thermal_stop",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "stage": stage,
                "progress": progress,
                **sample,
            }
            _append_jsonl(self.event_path, event)
            raise ThermalStop(
                f"GPU reached {sample['temperature_c']:.0f}C; "
                "latest immutable checkpoint/prefix is preserved"
            )
        if sample["temperature_c"] < self.policy.pause_at_c:
            return sample

        started = datetime.now(UTC)
        initial = sample
        print(
            "thermal_pause "
            f"stage={stage} progress={progress} "
            f"temperature_c={sample['temperature_c']:.0f} "
            f"resume_at_c={self.policy.resume_at_c}",
            flush=True,
        )
        while sample["temperature_c"] > self.policy.resume_at_c:
            time.sleep(self.policy.poll_seconds)
            sample = self._sample()
            if sample["temperature_c"] >= self.policy.hard_stop_at_c:
                _append_jsonl(
                    self.event_path,
                    {
                        "event": "hard_thermal_stop_during_cooldown",
                        "recorded_at_utc": datetime.now(UTC).isoformat(),
                        "stage": stage,
                        "progress": progress,
                        **sample,
                    },
                )
                raise ThermalStop(f"GPU reached {sample['temperature_c']:.0f}C during cooldown")
        _append_jsonl(
            self.event_path,
            {
                "event": "thermal_pause",
                "started_at_utc": started.isoformat(),
                "resumed_at_utc": datetime.now(UTC).isoformat(),
                "stage": stage,
                "progress": progress,
                "initial_sample": initial,
                "resume_sample": sample,
            },
        )
        print(
            "thermal_resume "
            f"stage={stage} progress={progress} "
            f"temperature_c={sample['temperature_c']:.0f}",
            flush=True,
        )
        return sample


def _resource_preflight(
    *,
    execution: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    resources = execution["resources"]
    gpu = _gpu_sample()
    memory_gib = _free_system_memory_gib()
    disk_gib = _free_disk_gib(output_root.parent)
    if memory_gib < float(resources["min_free_system_memory_gib"]):
        raise RuntimeError("insufficient free system memory for formal job")
    if disk_gib < float(resources["min_free_disk_gib"]):
        raise RuntimeError("insufficient free disk for immutable checkpoints")
    if gpu["memory_used_mib"] >= 512:
        raise RuntimeError("GPU is not free enough to start a formal job")
    return {
        "gpu": gpu,
        "free_system_memory_gib": memory_gib,
        "free_disk_gib": disk_gib,
    }


def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.no_grad()
def _mean_token_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    weighted_loss = 0.0
    shifted_supervised_tokens = 0
    for batch in loader:
        batch = _to_device(batch, device)
        token_count = int((batch["labels"][..., 1:] != -100).sum().item())
        loss = model(**batch).loss
        weighted_loss += float(loss.item()) * token_count
        shifted_supervised_tokens += token_count
    if shifted_supervised_tokens == 0:
        raise ValueError("validation has zero shifted supervised tokens")
    return weighted_loss / shifted_supervised_tokens, shifted_supervised_tokens


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    next_micro_batch_index: int,
    optimizer_steps: int,
    supervised_tokens_seen: int,
    epoch_weighted_loss: list[float],
    epoch_shifted_tokens: list[int],
    epoch_metrics: list[dict[str, Any]],
    pre_validation_loss: float,
    validation_tokens: int,
    training_seconds_completed: float,
    max_peak_memory_bytes_seen: int,
) -> dict[str, Any]:
    adapter_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in get_peft_model_state_dict(model).items()
    }
    return {
        "adapter_state": adapter_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": _rng_state(),
        "next_micro_batch_index": next_micro_batch_index,
        "optimizer_steps": optimizer_steps,
        "supervised_tokens_seen": supervised_tokens_seen,
        "epoch_weighted_loss": epoch_weighted_loss,
        "epoch_shifted_tokens": epoch_shifted_tokens,
        "epoch_metrics": epoch_metrics,
        "pre_validation_loss": pre_validation_loss,
        "validation_tokens": validation_tokens,
        "training_seconds_completed": training_seconds_completed,
        "max_peak_memory_bytes_seen": max_peak_memory_bytes_seen,
        "pending_micro_batches": 0,
    }


def _prepare_training_data(
    *,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    selected: list[dict[str, Any]],
    data_manifest_dir: Path,
    tokenizer: Any,
) -> tuple[
    list[dict[str, list[int]]],
    list[dict[str, Any]],
    DataLoader,
    list[dict[str, Any]],
]:
    training = recipe["training"]
    candidate_config = protocol["datasets"]["candidate_pool"]
    gsm_config = protocol["datasets"]["gsm8k"]
    tulu = load_dataset(
        candidate_config["repo_id"],
        candidate_config["config"],
        split="train",
        revision=candidate_config["revision"],
    )
    train_examples: list[dict[str, list[int]]] = []
    token_audit: list[dict[str, Any]] = []
    for candidate in selected:
        example, audit = tokenize_tulu_candidate(
            tokenizer=tokenizer,
            candidate=candidate,
            raw_row=tulu[int(candidate["source_index"])],
            max_length=int(training["max_length"]),
        )
        if (
            audit["total_tokens"] != candidate["total_tokens"]
            or audit["supervised_tokens"] != candidate["supervised_tokens"]
        ):
            raise ValueError(f"token audit changed for {candidate['candidate_id']}")
        train_examples.append(example)
        token_audit.append(audit)

    gsm_train = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        split="train",
        revision=gsm_config["revision"],
    )
    all_records = read_jsonl(data_manifest_dir / "gsm8k_records.jsonl")
    development_records = sorted(
        (row for row in all_records if row["protocol_split"] == "development"),
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    development_rows = [gsm_train[int(record["source_index"])] for record in development_records]
    for record, row in zip(
        development_records,
        development_rows,
        strict=True,
    ):
        validate_gsm8k_source_row(record, row)
    development_examples, development_audit = build_tokenized_overfit_examples(
        tokenizer=tokenizer,
        rows=development_rows,
        record_ids=[row["record_id"] for row in development_records],
        max_length=int(training["max_length"]),
    )
    collator = ResponseOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))
    development_loader = DataLoader(
        development_examples,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
    )
    return (
        train_examples,
        token_audit,
        development_loader,
        development_audit,
    )


def _build_trainable_model(
    *,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    model_config = protocol["model"]
    training = recipe["training"]
    model = AutoModelForCausalLM.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
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


def _finalize_training_artifacts(
    *,
    run_dir: Path,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    development_loader: DataLoader,
    token_audit: list[dict[str, Any]],
    development_audit: list[dict[str, Any]],
    parameter_report: Any,
    state: dict[str, Any],
    device: torch.device,
) -> Path:
    final_dir = run_dir / "training_complete"
    if final_dir.exists():
        return final_dir

    post_validation_loss, _ = _mean_token_loss(
        model,
        development_loader,
        device,
    )
    attempt = run_dir / f"training_complete_attempt_{uuid.uuid4().hex}"
    attempt.mkdir(parents=False, exist_ok=False)
    adapter_dir = attempt / "adapter"
    tokenizer_dir = attempt / "tokenizer"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    adapter_sha256 = file_sha256(adapter_dir / "adapter_model.safetensors")

    model.to("cpu")
    _optimizer_to_device(optimizer, torch.device("cpu"))
    del scheduler, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    model_config = protocol["model"]
    reloaded_base = AutoModelForCausalLM.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    reloaded_model = PeftModel.from_pretrained(
        reloaded_base,
        adapter_dir,
        is_trainable=False,
    ).to(device)
    reloaded_validation_loss, _ = _mean_token_loss(
        reloaded_model,
        development_loader,
        device,
    )
    reload_difference = abs(reloaded_validation_loss - post_validation_loss)
    del reloaded_model, reloaded_base
    gc.collect()
    torch.cuda.empty_cache()

    training = recipe["training"]
    total_micro_batches = len(token_audit) * int(training["epochs"])
    accumulation = int(training["gradient_accumulation_steps"])
    optimizer_steps_planned = math.ceil(total_micro_batches / accumulation)
    metrics = {
        "status": "PASS",
        "strategy": _read_json(run_dir / "manifest.json")["config"]["strategy"],
        "seed": _read_json(run_dir / "manifest.json")["seed"],
        "selected_count": len(token_audit),
        "pre_validation_token_loss": state["pre_validation_loss"],
        "post_validation_token_loss": post_validation_loss,
        "reloaded_validation_token_loss": reloaded_validation_loss,
        "adapter_reload_loss_absolute_difference": reload_difference,
        "adapter_reload_gate_difference_at_most_1e_6": (reload_difference <= 1e-6),
        "validation_shifted_supervised_tokens": state["validation_tokens"],
        "epochs": int(training["epochs"]),
        "optimizer_steps_planned": optimizer_steps_planned,
        "optimizer_steps_completed": state["optimizer_steps"],
        "supervised_tokens_seen": state["supervised_tokens_seen"],
        "training_seconds_checkpointed": state["training_seconds_completed"],
        "supervised_tokens_per_second_checkpointed": (
            state["supervised_tokens_seen"] / state["training_seconds_completed"]
        ),
        "peak_training_memory_bytes": state["max_peak_memory_bytes_seen"],
        "peak_training_memory_gib": (state["max_peak_memory_bytes_seen"] / 1024**3),
        "trainable_parameters": parameter_report.trainable_parameters,
        "total_parameters": parameter_report.total_parameters,
        "trainable_fraction": (
            parameter_report.trainable_parameters / parameter_report.total_parameters
        ),
        "adapter_model_sha256": adapter_sha256,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "This is one formal random seed-17 run. It does not establish "
            "a selector comparison or seed variance."
        ),
    }
    _write_json_exclusive(attempt / "training_metrics.json", metrics)
    _write_json_exclusive(
        attempt / "epoch_metrics.json",
        state["epoch_metrics"],
    )
    _write_json_exclusive(
        attempt / "training_token_audit.json",
        token_audit,
    )
    _write_json_exclusive(
        attempt / "development_token_audit.json",
        development_audit,
    )
    _write_json_exclusive(
        attempt / "artifact_manifest.json",
        {
            "adapter_model_sha256": adapter_sha256,
            "training_metrics_sha256": file_sha256(attempt / "training_metrics.json"),
            "epoch_metrics_sha256": file_sha256(attempt / "epoch_metrics.json"),
            "training_token_audit_sha256": file_sha256(attempt / "training_token_audit.json"),
            "development_token_audit_sha256": file_sha256(attempt / "development_token_audit.json"),
        },
    )
    attempt.rename(final_dir)
    return final_dir


def _train(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    execution: dict[str, Any],
    selected: list[dict[str, Any]],
    data_manifest_dir: Path,
    thermal: ThermalController,
) -> Path:
    final_dir = run_dir / "training_complete"
    if final_dir.is_dir():
        return final_dir

    set_seed(int(manifest["seed"]))
    device = torch.device("cuda")
    model_config = protocol["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    (
        train_examples,
        token_audit,
        development_loader,
        development_audit,
    ) = _prepare_training_data(
        protocol=protocol,
        recipe=recipe,
        selected=selected,
        data_manifest_dir=data_manifest_dir,
        tokenizer=tokenizer,
    )
    collator = ResponseOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))
    thermal.wait_until_start_safe()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = _build_trainable_model(
        protocol=protocol,
        recipe=recipe,
        device=device,
    )
    parameter_report = audit_lora_parameters(model)
    training = recipe["training"]
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    total_micro_batches = len(train_examples) * epochs
    optimizer_steps_planned = math.ceil(total_micro_batches / accumulation)
    warmup_steps = math.ceil(optimizer_steps_planned * float(training["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps_planned,
    )
    checkpoint_binding = {
        "run_config_hash": manifest["config_hash"],
        "git_commit": manifest["git_commit"],
        "strategy": manifest["config"]["strategy"],
        "seed": manifest["seed"],
        "selected_id_sha256": manifest["config"]["selected_id_sha256"],
    }
    checkpoint_dir = run_dir / "checkpoints"
    latest = load_latest_checkpoint(
        checkpoint_directory=checkpoint_dir,
        expected_binding=checkpoint_binding,
    )
    if latest is None:
        pre_validation_loss, validation_tokens = _mean_token_loss(
            model,
            development_loader,
            device,
        )
        state = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            next_micro_batch_index=0,
            optimizer_steps=0,
            supervised_tokens_seen=0,
            epoch_weighted_loss=[0.0 for _ in range(epochs)],
            epoch_shifted_tokens=[0 for _ in range(epochs)],
            epoch_metrics=[],
            pre_validation_loss=pre_validation_loss,
            validation_tokens=validation_tokens,
            training_seconds_completed=0.0,
            max_peak_memory_bytes_seen=int(torch.cuda.max_memory_allocated()),
        )
        saved = write_immutable_checkpoint(
            checkpoint_directory=checkpoint_dir,
            state=state,
            binding=checkpoint_binding,
        )
        _append_jsonl(
            run_dir / "runtime_events.jsonl",
            {
                "event": "checkpoint_saved",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                **saved["sidecar"],
            },
        )
    else:
        state, sidecar = latest
        set_peft_model_state_dict(model, state["adapter_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state"])
        _restore_rng_state(state["rng_state"])
        _append_jsonl(
            run_dir / "runtime_events.jsonl",
            {
                "event": "checkpoint_resumed",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                **sidecar,
            },
        )

    next_micro_batch = int(state["next_micro_batch_index"])
    optimizer_steps = int(state["optimizer_steps"])
    supervised_tokens_seen = int(state["supervised_tokens_seen"])
    epoch_weighted_loss = list(state["epoch_weighted_loss"])
    epoch_shifted_tokens = list(state["epoch_shifted_tokens"])
    epoch_metrics = list(state["epoch_metrics"])
    pre_validation_loss = float(state["pre_validation_loss"])
    validation_tokens = int(state["validation_tokens"])
    checkpointed_seconds = float(state["training_seconds_completed"])
    max_peak_memory = int(state["max_peak_memory_bytes_seen"])
    if int(state["pending_micro_batches"]) != 0:
        raise ValueError("resumable checkpoint must be at an optimizer boundary")

    orders = deterministic_epoch_orders(
        example_count=len(train_examples),
        epochs=epochs,
        seed=int(manifest["seed"]),
    )
    optimizer.zero_grad(set_to_none=True)
    pending_micro_batches = 0
    gradient_audited = optimizer_steps > 0
    invocation_started = time.perf_counter()
    model.train()
    for flat_index in range(next_micro_batch, total_micro_batches):
        if flat_index % thermal.policy.training_check_every_micro_batches == 0:
            thermal.guard(stage="training_before", progress=flat_index)
        epoch = flat_index // len(train_examples)
        position = flat_index % len(train_examples)
        example_index = orders[epoch][position]
        batch = collator([train_examples[example_index]])
        batch = _to_device(batch, device)
        token_count = int((batch["labels"][..., 1:] != -100).sum().item())
        loss = model(**batch).loss
        (loss / accumulation).backward()
        pending_micro_batches += 1
        if not gradient_audited:
            audit_lora_gradients(model)
            gradient_audited = True
        epoch_weighted_loss[epoch] += float(loss.detach().item()) * token_count
        epoch_shifted_tokens[epoch] += token_count
        supervised_tokens_seen += token_count

        at_epoch_end = position + 1 == len(train_examples)
        if at_epoch_end and len(epoch_metrics) <= epoch:
            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "train_token_loss": (epoch_weighted_loss[epoch] / epoch_shifted_tokens[epoch]),
                    "shifted_supervised_tokens": epoch_shifted_tokens[epoch],
                    "optimizer_steps_completed_before_final_remainder": (optimizer_steps),
                }
            )
            print(
                f"epoch={epoch + 1}/{epochs} "
                "train_token_loss="
                f"{epoch_metrics[-1]['train_token_loss']:.6f} "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )

        should_step = pending_micro_batches == accumulation
        final_micro_batch = flat_index + 1 == total_micro_batches
        if should_step or final_micro_batch:
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                float(training["gradient_clipping"]),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending_micro_batches = 0
            max_peak_memory = max(
                max_peak_memory,
                int(torch.cuda.max_memory_allocated()),
            )
            elapsed = checkpointed_seconds + time.perf_counter() - invocation_started
            state = _checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                next_micro_batch_index=flat_index + 1,
                optimizer_steps=optimizer_steps,
                supervised_tokens_seen=supervised_tokens_seen,
                epoch_weighted_loss=epoch_weighted_loss,
                epoch_shifted_tokens=epoch_shifted_tokens,
                epoch_metrics=epoch_metrics,
                pre_validation_loss=pre_validation_loss,
                validation_tokens=validation_tokens,
                training_seconds_completed=elapsed,
                max_peak_memory_bytes_seen=max_peak_memory,
            )
            saved = write_immutable_checkpoint(
                checkpoint_directory=checkpoint_dir,
                state=state,
                binding=checkpoint_binding,
            )
            _append_jsonl(
                run_dir / "runtime_events.jsonl",
                {
                    "event": "checkpoint_saved",
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                    **saved["sidecar"],
                },
            )
            checkpointed_seconds = elapsed
            invocation_started = time.perf_counter()
            print(
                "training_progress "
                f"micro_batches={flat_index + 1}/{total_micro_batches} "
                f"optimizer_steps={optimizer_steps}/{optimizer_steps_planned}",
                flush=True,
            )

        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        if peak_gib > float(execution["resources"]["max_peak_gpu_memory_gib"]):
            raise RuntimeError(f"peak GPU allocation {peak_gib:.3f} GiB exceeded guard")
        thermal.guard(stage="training_after", progress=flat_index + 1)
        time.sleep(thermal.policy.training_inter_micro_batch_sleep_seconds)

    if pending_micro_batches:
        raise AssertionError("final remainder was not committed")
    if optimizer_steps != optimizer_steps_planned:
        raise ValueError("completed optimizer steps do not match frozen plan")
    return _finalize_training_artifacts(
        run_dir=run_dir,
        protocol=protocol,
        recipe=recipe,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        development_loader=development_loader,
        token_audit=token_audit,
        development_audit=development_audit,
        parameter_report=parameter_report,
        state=state,
        device=device,
    )


def _ensure_evaluation_manifest(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    protocol_path: Path,
    recipe_path: Path,
    execution_path: Path,
    adapter_path: Path,
    recipe: dict[str, Any],
) -> Path:
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    path = evaluation_dir / "manifest.json"
    payload = {
        "evaluation_schema_version": "b500-formal-resumable-eval-v1",
        "source_run_id": manifest["run_id"],
        "source_run_git_commit": manifest["git_commit"],
        "evaluation_code_git_commit": _git_commit(),
        "protocol_config_sha256": file_sha256(protocol_path),
        "recipe_config_sha256": file_sha256(recipe_path),
        "execution_policy_sha256": file_sha256(execution_path),
        "adapter_model_sha256": file_sha256(adapter_path),
        "training_seed": manifest["seed"],
        "strategy": manifest["config"]["strategy"],
        "semantic_evaluation": recipe["evaluation"],
    }
    if path.exists():
        if _read_json(path) != payload:
            raise ValueError("evaluation manifest changed after evaluation began")
    else:
        _write_json_exclusive(path, payload)
    return evaluation_dir


def _evaluate(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    protocol_path: Path,
    recipe_path: Path,
    execution_path: Path,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    data_manifest_dir: Path,
    training_complete: Path,
    thermal: ThermalController,
) -> Path:
    evaluation = recipe["evaluation"]
    if evaluation["prompt_version"] != PROMPT_VERSION:
        raise ValueError("frozen evaluation prompt changed")
    adapter_path = training_complete / "adapter" / "adapter_model.safetensors"
    evaluation_dir = _ensure_evaluation_manifest(
        run_dir=run_dir,
        manifest=manifest,
        protocol_path=protocol_path,
        recipe_path=recipe_path,
        execution_path=execution_path,
        adapter_path=adapter_path,
        recipe=recipe,
    )
    raw_path = evaluation_dir / "raw_outputs.jsonl"
    metrics_path = evaluation_dir / "metrics.json"
    all_records = read_jsonl(data_manifest_dir / "gsm8k_records.jsonl")
    records = sorted(
        (row for row in all_records if row["protocol_split"] == evaluation["split"]),
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    if len(records) != int(evaluation["example_count"]):
        raise ValueError("held-out test count changed")
    completed = read_jsonl(raw_path) if raw_path.exists() else []
    next_index = validate_completed_prefix(
        completed_rows=completed,
        frozen_records=records,
    )
    if metrics_path.exists():
        if next_index != len(records):
            raise ValueError("metrics exist before evaluation is complete")
        return evaluation_dir

    set_seed(int(manifest["seed"]))
    model_config = protocol["model"]
    gsm_config = protocol["datasets"]["gsm8k"]
    tokenizer = AutoTokenizer.from_pretrained(
        training_complete / "tokenizer",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    thermal.wait_until_start_safe()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        training_complete / "adapter",
        is_trainable=False,
    ).to("cuda")
    model.config.use_cache = True
    model.eval()
    gsm_test = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        split="test",
        revision=gsm_config["revision"],
    )
    invocation_started = time.perf_counter()
    mode = "a" if raw_path.exists() else "x"
    with raw_path.open(mode, encoding="utf-8", newline="\n") as output:
        for index in range(next_index, len(records)):
            if index % thermal.policy.evaluation_check_every_examples == 0:
                thermal.guard(stage="evaluation_before", progress=index)
            record = records[index]
            row = gsm_test[int(record["source_index"])]
            validate_gsm8k_source_row(record, row)
            prompt = build_evaluation_prompt(row["question"])
            encoded = tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(evaluation["max_input_length"]),
            ).to("cuda")
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=bool(evaluation["do_sample"]),
                    num_beams=int(evaluation["num_beams"]),
                    max_new_tokens=int(evaluation["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            token_ids = generated[0, input_width:]
            raw_output = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()
            scored = score_generation(
                record=record,
                gold_answer_text=row["answer"],
                generated_text=raw_output,
            )
            output.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            if (index + 1) % 10 == 0 or index + 1 == len(records):
                print(
                    f"evaluation={index + 1}/{len(records)}",
                    flush=True,
                )
            thermal.guard(
                stage="evaluation_after",
                progress=index + 1,
            )
            time.sleep(thermal.policy.evaluation_inter_example_sleep_seconds)
    torch.cuda.synchronize()
    rows = read_jsonl(raw_path)
    validate_completed_prefix(
        completed_rows=rows,
        frozen_records=records,
    )
    metrics = {
        **aggregate_gsm8k_metrics(rows),
        "strategy": manifest["config"]["strategy"],
        "seed": manifest["seed"],
        "source_run_id": manifest["run_id"],
        "adapter_model_sha256": file_sha256(adapter_path),
        "raw_outputs_sha256": file_sha256(raw_path),
        "evaluation_seconds_this_invocation": (time.perf_counter() - invocation_started),
        "peak_evaluation_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_evaluation_memory_gib": (torch.cuda.max_memory_allocated() / 1024**3),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "This is the first of nine preregistered formal jobs. No "
            "cross-strategy or seed-level conclusion is permitted yet."
        ),
    }
    _write_json_exclusive(metrics_path, metrics)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return evaluation_dir


def _complete_run(
    *,
    run_dir: Path,
    training_complete: Path,
    evaluation_dir: Path,
    thermal: ThermalController,
) -> None:
    completion_path = run_dir / "run_complete.json"
    if completion_path.exists():
        return
    training_metrics = training_complete / "training_metrics.json"
    evaluation_metrics = evaluation_dir / "metrics.json"
    _write_json_exclusive(
        completion_path,
        {
            "status": "PASS",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "training_metrics_sha256": file_sha256(training_metrics),
            "evaluation_metrics_sha256": file_sha256(evaluation_metrics),
            "raw_outputs_sha256": file_sha256(evaluation_dir / "raw_outputs.jsonl"),
            "adapter_model_sha256": file_sha256(
                training_complete / "adapter" / "adapter_model.safetensors"
            ),
            "max_temperature_c_observed_by_runner": (thermal.max_temperature_c),
            "next_job_started": False,
        },
    )


def _resolve_contract(
    *,
    matrix_path: Path,
    strategy: str,
    seed: int,
) -> dict[str, Any]:
    spec = _read_json(matrix_path)
    report = preflight_b500_matrix(
        spec=spec,
        repo_root=ROOT,
        matrix_config_path=matrix_path.relative_to(ROOT).as_posix(),
    )
    if report["status"] != "READY_FOR_MANUAL_ONE_JOB_AT_A_TIME":
        raise RuntimeError("formal matrix preflight is not ready")
    matching = [
        job for job in report["jobs"] if job["strategy"] == strategy and int(job["seed"]) == seed
    ]
    if len(matching) != 1:
        raise ValueError("requested strategy/seed is not one matrix job")
    if matching[0]["status"] != "READY_FOR_MANUAL_INVOCATION":
        raise RuntimeError("requested matrix job is not ready")
    paths = {
        "protocol": (ROOT / spec["protocol_config"]["path"]).resolve(),
        "recipe": (ROOT / spec["recipe_config"]["path"]).resolve(),
        "execution": (ROOT / spec["execution_config"]["path"]).resolve(),
        "selection": (ROOT / spec["selections"][strategy]["path"]).resolve(),
        "data": (ROOT / spec["data_manifest"]["directory"]).resolve(),
        "output": (ROOT / spec["output_root"]).resolve(),
    }
    return {
        "spec": spec,
        "report": report,
        "job": matching[0],
        "paths": paths,
    }


def _create_or_resume_run(
    *,
    contract: dict[str, Any],
    strategy: str,
    seed: int,
    resume_run_dir: Path | None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    paths = contract["paths"]
    protocol = _read_json(paths["protocol"])
    recipe = _read_json(paths["recipe"])
    execution = _read_json(paths["execution"])
    validate_execution_policy(execution)
    selection_manifest = _read_json(paths["selection"])
    selected = validate_selection_manifest(
        selection_manifest,
        expected_strategy=strategy,
        expected_budget=int(recipe["selection"]["budget"]),
        expected_selection_seed=int(recipe["selection"]["selection_seed"]),
    )
    if selected_id_sha256(selected) != selection_manifest["selected_id_sha256"]:
        raise ValueError("selected candidate ID hash mismatch")
    run_config = {
        "matrix_version": contract["spec"]["matrix_version"],
        "matrix_config_sha256": contract["report"]["matrix_config_sha256"],
        "common_contract_sha256": contract["report"]["common_contract_sha256"],
        "protocol_version": recipe["protocol_version"],
        "strategy": strategy,
        "selection_manifest_sha256": file_sha256(paths["selection"]),
        "selected_id_sha256": selection_manifest["selected_id_sha256"],
        "model": protocol["model"],
        "datasets": protocol["datasets"],
        "training": recipe["training"],
        "evaluation": recipe["evaluation"],
        "execution_policy_sha256": file_sha256(paths["execution"]),
        "execution_policy": execution,
    }
    output_root = paths["output"]
    if resume_run_dir is None:
        if output_root.is_dir():
            for manifest_path in output_root.glob("*/manifest.json"):
                existing = _read_json(manifest_path)
                if (
                    existing.get("seed") == seed
                    and existing.get("config", {}).get("strategy") == strategy
                ):
                    raise FileExistsError(
                        "this strategy/seed already exists; use --resume-run-dir explicitly"
                    )
        gsm_config = protocol["datasets"]["gsm8k"]
        candidate_config = protocol["datasets"]["candidate_pool"]
        run_dir, manifest = create_run_manifest(
            output_root=output_root,
            repo_root=ROOT,
            stage=f"b500_formal_{strategy}",
            config=run_config,
            seed=seed,
            command=[sys.executable, *sys.argv],
            dataset_revisions={
                gsm_config["repo_id"]: gsm_config["revision"],
                candidate_config["repo_id"]: candidate_config["revision"],
            },
            model_revision=protocol["model"]["revision"],
            extra={
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
            },
        )
    else:
        run_dir = resume_run_dir.resolve()
        try:
            run_dir.relative_to(output_root)
        except ValueError as error:
            raise ValueError("resume directory is outside formal output root") from error
        manifest = _read_json(run_dir / "manifest.json")
        if manifest["config"] != run_config:
            raise ValueError("resume run contract differs from frozen matrix")
        if manifest["seed"] != seed:
            raise ValueError("resume run seed differs")
        if manifest["git_commit"] != _git_commit():
            raise ValueError("resume must use the original committed code")
    return run_dir, manifest, protocol, recipe, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=Path("configs/b500_formal_matrix_v1.json"),
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "rds_all", "rds_error"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the exact job and resources without creating outputs.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("formal B=500 run requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal B=500 run requires BF16 support")
    _require_clean_git_worktree()
    contract = _resolve_contract(
        matrix_path=args.matrix_config.resolve(),
        strategy=args.strategy,
        seed=args.seed,
    )
    execution = _read_json(contract["paths"]["execution"])
    policy = validate_execution_policy(execution)
    resources = _resource_preflight(
        execution=execution,
        output_root=contract["paths"]["output"],
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "job_id": contract["job"]["job_id"],
                    "automatic_execution": False,
                    "resources": resources,
                    "execution_policy_sha256": file_sha256(contract["paths"]["execution"]),
                    "selection_manifest_sha256": file_sha256(contract["paths"]["selection"]),
                    "output_root_exists": contract["paths"]["output"].exists(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    torch.set_num_threads(int(execution["resources"]["cpu_threads"]))
    output_root = contract["paths"]["output"]
    with _global_job_lock(output_root):
        run_dir, manifest, protocol, recipe, selected = _create_or_resume_run(
            contract=contract,
            strategy=args.strategy,
            seed=args.seed,
            resume_run_dir=args.resume_run_dir,
        )
        thermal = ThermalController(
            policy=policy,
            event_path=run_dir / "thermal_events.jsonl",
        )
        _append_jsonl(
            run_dir / "invocations.jsonl",
            {
                "event": "invocation_start",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "command": [sys.executable, *sys.argv],
                "git_commit": _git_commit(),
                "resources": resources,
            },
        )
        try:
            training_complete = _train(
                run_dir=run_dir,
                manifest=manifest,
                protocol=protocol,
                recipe=recipe,
                execution=execution,
                selected=selected,
                data_manifest_dir=contract["paths"]["data"],
                thermal=thermal,
            )
            evaluation_dir = _evaluate(
                run_dir=run_dir,
                manifest=manifest,
                protocol_path=contract["paths"]["protocol"],
                recipe_path=contract["paths"]["recipe"],
                execution_path=contract["paths"]["execution"],
                protocol=protocol,
                recipe=recipe,
                data_manifest_dir=contract["paths"]["data"],
                training_complete=training_complete,
                thermal=thermal,
            )
            _complete_run(
                run_dir=run_dir,
                training_complete=training_complete,
                evaluation_dir=evaluation_dir,
                thermal=thermal,
            )
        except ThermalStop as error:
            _append_jsonl(
                run_dir / "invocations.jsonl",
                {
                    "event": "invocation_thermal_stop",
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                    "error": str(error),
                    "max_temperature_c_observed_by_runner": (thermal.max_temperature_c),
                },
            )
            print(
                json.dumps(
                    {
                        "status": "THERMAL_STOP",
                        "run_dir": str(run_dir),
                        "message": str(error),
                        "resume_required": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            raise SystemExit(75) from error
        except Exception as error:
            _append_jsonl(
                run_dir / "failures.jsonl",
                {
                    "event": "unexpected_failure",
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        _append_jsonl(
            run_dir / "invocations.jsonl",
            {
                "event": "invocation_complete",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "max_temperature_c_observed_by_runner": (thermal.max_temperature_c),
            },
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "run_dir": str(run_dir),
                    "next_job_started": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
