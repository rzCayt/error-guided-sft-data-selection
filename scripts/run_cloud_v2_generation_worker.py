"""Run one resumable, batch-one shard for same-GPU concurrency calibration."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import (  # noqa: E402
    _append_jsonl,
    _read_json,
    _require_clean_git_worktree,
    _write_json_exclusive,
)

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.cloud_v2_batching import append_jsonl_rows_fsynced  # noqa: E402
from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.evaluation.two_worker_calibration import (  # noqa: E402
    records_for_shard,
    validate_two_worker_config,
    validate_worker_prefix,
)
from eg_sft.experiment.cloud_v2_calibration import (  # noqa: E402
    read_json_object,
    repository_path,
    resolve_frozen_artifact,
    validate_calibration_config,
)
from eg_sft.experiment.cloud_v2_train_runtime import validate_data_bindings  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _gpu_uuid(device_index: int) -> str:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for row in process.stdout.splitlines():
        index, separator, uuid = row.partition(",")
        if separator and int(index.strip()) == device_index:
            return uuid.strip()
    raise RuntimeError(f"nvidia-smi did not report CUDA device index {device_index}")


def _load_context(config_path: Path, shard_id: str) -> dict[str, Any]:
    config = read_json_object(config_path)
    shards = validate_two_worker_config(config)
    matching = [shard for shard in shards if shard.shard_id == shard_id]
    if len(matching) != 1:
        raise ValueError("shard_id is not in the frozen two-worker config")
    base_path = resolve_frozen_artifact(
        repo_root=ROOT,
        binding=config["base_calibration_config"],
        label="base calibration config",
    )
    base_config = read_json_object(base_path)
    validate_calibration_config(base_config)
    protocol_path = resolve_frozen_artifact(
        repo_root=ROOT,
        binding=base_config["protocol_config"],
        label="protocol config",
    )
    protocol = _read_json(protocol_path)
    data_manifest_dir = validate_data_bindings(repo_root=ROOT, config=base_config)
    return {
        "config": config,
        "shard": matching[0],
        "base_config": base_config,
        "base_config_path": base_path,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "data_manifest_dir": data_manifest_dir,
    }


def _validated_training_artifact(
    *,
    training_run_dir: Path,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    training_root = repository_path(
        ROOT,
        str(base_config["training_output_root"]),
        label="training calibration output root",
    )
    training_run_dir.relative_to(training_root)
    manifest = _read_json(training_run_dir / "manifest.json")
    metrics = _read_json(training_run_dir / "training_complete" / "calibration_metrics.json")
    if metrics.get("status") != "PASS":
        raise ValueError("two-worker calibration requires a passed training adapter")
    if manifest.get("config", {}).get("entry_point") != (
        "scripts/run_b500_cloud_v2_train_calibration_fixed.py"
    ):
        raise ValueError("training adapter did not come from the fixed calibration entry")
    adapter_dir = training_run_dir / "training_complete" / "adapter"
    tokenizer_dir = training_run_dir / "training_complete" / "tokenizer"
    return {
        "manifest": manifest,
        "metrics": metrics,
        "adapter_dir": adapter_dir,
        "tokenizer_dir": tokenizer_dir,
        "adapter_sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher-run-dir", type=Path, required=True)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud_v2_two_worker_generation_v1.json"),
    )
    parser.add_argument("--shard-id", choices=["shard0", "shard1"], required=True)
    args = parser.parse_args()

    invocation_started = time.perf_counter()
    model_load_seconds = 0.0
    generation_seconds = 0.0
    worker_dir: Path | None = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("two-worker generation requires a BF16 CUDA GPU")
        _require_clean_git_worktree()
        context = _load_context(args.config.resolve(), args.shard_id)
        config = context["config"]
        shard = context["shard"]
        launcher_run_dir = args.launcher_run_dir.resolve()
        output_root = repository_path(
            ROOT,
            str(config["output_root"]),
            label="two-worker output root",
        )
        launcher_run_dir.relative_to(output_root)
        launcher_manifest = _read_json(launcher_run_dir / "launcher_manifest.json")
        training_run_dir = args.training_run_dir.resolve()
        training = _validated_training_artifact(
            training_run_dir=training_run_dir,
            base_config=context["base_config"],
        )
        if launcher_manifest["adapter_model_sha256"] != training["adapter_sha256"]:
            raise ValueError("launcher and worker adapter hashes differ")
        device_index = int(config["cuda_device_index"])
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        gpu_uuid = _gpu_uuid(device_index)
        worker_dir = launcher_run_dir / "workers" / shard.shard_id
        worker_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = worker_dir / "manifest.json"
        worker_manifest = {
            "worker_schema_version": "cloud-v2-two-worker-shard-v1",
            "launcher_run_id": launcher_manifest["run_id"],
            "worker": {
                "shard_id": shard.shard_id,
                "start_index": shard.start_index,
                "end_index": shard.end_index,
                "physical_batch_size": 1,
                "cuda_device_index": device_index,
            },
            "gpu_uuid": gpu_uuid,
            "adapter_model_sha256": training["adapter_sha256"],
            "source_training_run_id": training["manifest"]["run_id"],
            "two_worker_config_sha256": file_sha256(args.config.resolve()),
            "base_calibration_config_sha256": file_sha256(context["base_config_path"]),
            "protocol_config_sha256": file_sha256(context["protocol_path"]),
        }
        if manifest_path.exists():
            if _read_json(manifest_path) != worker_manifest:
                raise ValueError("worker manifest changed after shard execution began")
        else:
            _write_json_exclusive(manifest_path, worker_manifest)
        metrics_path = worker_dir / "metrics.json"
        raw_path = worker_dir / "raw_outputs.jsonl"
        all_records = read_jsonl(context["data_manifest_dir"] / "gsm8k_records.jsonl")
        records = sorted(
            (
                row
                for row in all_records
                if row["protocol_split"] == config["generation_protocol_split"]
            ),
            key=lambda row: (row["source_index"], row["record_id"]),
        )
        shard_records = records_for_shard(records, shard)
        completed = read_jsonl(raw_path) if raw_path.exists() else []
        next_offset = validate_worker_prefix(
            rows=completed,
            frozen_shard_records=shard_records,
            shard_id=shard.shard_id,
        )
        if metrics_path.exists():
            if next_offset != shard.count:
                raise ValueError("worker metrics exist before its shard is complete")
            print(metrics_path.read_text(encoding="utf-8"))
            return

        set_seed(int(context["base_config"]["training_seed"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(training["tokenizer_dir"], use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            context["protocol"]["model"]["repo_id"],
            revision=context["protocol"]["model"]["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=str(context["base_config"]["attention_implementation"]),
        )
        model = PeftModel.from_pretrained(
            base,
            training["adapter_dir"],
            is_trainable=False,
        ).to(device)
        model.config.use_cache = True
        model.eval()
        torch.cuda.synchronize(device)
        model_load_seconds = time.perf_counter() - load_started
        gsm_train = load_dataset(
            context["protocol"]["datasets"]["gsm8k"]["repo_id"],
            context["protocol"]["datasets"]["gsm8k"]["config"],
            split="train",
            revision=context["protocol"]["datasets"]["gsm8k"]["revision"],
        )
        generation_started = time.perf_counter()
        generation_config = config["generation"]
        for offset in range(next_offset, len(shard_records)):
            record = shard_records[offset]
            source_row = gsm_train[int(record["source_index"])]
            validate_gsm8k_source_row(record, source_row)
            prompt = build_evaluation_prompt(source_row["question"])
            encoded = tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(generation_config["max_input_length"]),
            ).to(device)
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=bool(generation_config["do_sample"]),
                    num_beams=int(generation_config["num_beams"]),
                    max_new_tokens=int(generation_config["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            raw_output = tokenizer.decode(
                generated[0, input_width:],
                skip_special_tokens=True,
            ).strip()
            scored = score_generation(
                record=record,
                gold_answer_text=source_row["answer"],
                generated_text=raw_output,
            )
            scored.update(
                {
                    "shard_id": shard.shard_id,
                    "shard_offset": offset,
                    "global_record_index": shard.start_index + offset,
                }
            )
            append_jsonl_rows_fsynced(raw_path, [scored])
            if (offset + 1) % 8 == 0 or offset + 1 == shard.count:
                print(
                    f"worker={shard.shard_id} progress={offset + 1}/{shard.count}",
                    flush=True,
                )
        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_started
        rows = read_jsonl(raw_path)
        validate_worker_prefix(
            rows=rows,
            frozen_shard_records=shard_records,
            shard_id=shard.shard_id,
        )
        if len(rows) != shard.count:
            raise ValueError("worker shard did not complete all 64 records")
        worker_wall_seconds = time.perf_counter() - invocation_started
        _append_jsonl(
            worker_dir / "invocations.jsonl",
            {
                "event": "worker_invocation_complete",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "model_load_seconds": model_load_seconds,
                "generation_seconds": generation_seconds,
                "worker_wall_seconds": worker_wall_seconds,
            },
        )
        invocation_rows = read_jsonl(worker_dir / "invocations.jsonl")
        completed_invocations = [
            row for row in invocation_rows if row.get("event") == "worker_invocation_complete"
        ]
        metrics = {
            "status": "PASS",
            "study_role": "same_gpu_concurrency_calibration_only_not_formal_training",
            "shard_id": shard.shard_id,
            "record_count": len(rows),
            "model_load_seconds": sum(
                float(row["model_load_seconds"]) for row in completed_invocations
            ),
            "generation_seconds": sum(
                float(row["generation_seconds"]) for row in completed_invocations
            ),
            "worker_wall_seconds": sum(
                float(row["worker_wall_seconds"]) for row in completed_invocations
            ),
            "resume_invocation_count": len(completed_invocations),
            "peak_allocated_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_memory_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "gpu_uuid": gpu_uuid,
            "adapter_model_sha256": training["adapter_sha256"],
            "raw_outputs_sha256": file_sha256(raw_path),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json_exclusive(metrics_path, metrics)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception as error:
        if worker_dir is not None:
            _append_jsonl(
                worker_dir / "failures.jsonl",
                {
                    "event": "worker_failure",
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "model_load_seconds_this_invocation": model_load_seconds,
                    "generation_seconds_this_invocation": generation_seconds,
                    "worker_wall_seconds_this_invocation": (
                        time.perf_counter() - invocation_started
                    ),
                },
            )
        raise


if __name__ == "__main__":
    main()
