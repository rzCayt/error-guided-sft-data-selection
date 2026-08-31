"""Run one resumable batch-one shard of a sealed formal GSM8K evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

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
from run_cloud_v2_generation_worker import _gpu_uuid  # noqa: E402

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.cloud_v2_batching import (  # noqa: E402
    append_jsonl_rows_fsynced,
    contiguous_record_batches,
)
from eg_sft.evaluation.formal_two_worker import (  # noqa: E402
    formal_shards,
    records_for_formal_shard,
    validate_formal_worker_prefix,
)
from eg_sft.evaluation.identifiable_batch_backend import (  # noqa: E402
    generated_token_rows,
    record_generated_token_ids,
    resolve_eval_batch_size,
)
from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    PROMPT_VERSION,
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.experiment.cloud_v2_formal import (  # noqa: E402
    resolve_formal_contract,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud_v2_formal_b500_single_cell_v1.json"),
    )
    parser.add_argument(
        "--shard-id",
        choices=["test_shard0", "test_shard1"],
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    physical_batch_size, identifiable_v4 = resolve_eval_batch_size(
        matrix_config=_read_json(config_path),
        environ=os.environ,
    )

    invocation_started = time.perf_counter()
    worker_dir: Path | None = None
    model_load_seconds = 0.0
    generation_seconds = 0.0
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("formal evaluation worker requires a BF16 CUDA GPU")
        _require_clean_git_worktree()
        run_dir = args.run_dir.resolve()
        run_manifest = _read_json(run_dir / "manifest.json")
        method = str(run_manifest["config"]["method"])
        seed = int(run_manifest["seed"])
        contract = resolve_formal_contract(
            repo_root=ROOT,
            config_path=config_path,
            method=method,
            seed=seed,
        )
        run_dir.relative_to(contract["output_root"])
        if run_manifest["git_commit"] != contract["config"].get(
            "expected_git_commit", run_manifest["git_commit"]
        ):
            raise ValueError("formal worker git binding changed")
        evaluation = contract["config"]["evaluation"]
        if evaluation["prompt_version"] != PROMPT_VERSION:
            raise ValueError("formal prompt version changed")
        shards = formal_shards(evaluation)
        shard = next(item for item in shards if item.shard_id == args.shard_id)
        training_dir = run_dir / "training_complete"
        training_metrics = _read_json(training_dir / "training_metrics.json")
        if training_metrics.get("status") != "PASS":
            raise ValueError("formal evaluation requires passed training artifacts")
        adapter_dir = training_dir / "adapter"
        adapter_path = adapter_dir / "adapter_model.safetensors"
        adapter_sha256 = file_sha256(adapter_path)
        if adapter_sha256 != training_metrics.get("adapter_model_sha256"):
            raise ValueError("formal adapter SHA-256 changed")
        device_index = int(evaluation["cuda_device_index"])
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        gpu_uuid = _gpu_uuid(device_index)
        worker_dir = run_dir / "evaluation" / "workers" / shard.shard_id
        worker_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = worker_dir / "manifest.json"
        worker_manifest = {
            "worker_schema_version": "cloud-v2-formal-eval-worker-v1",
            "source_run_id": run_manifest["run_id"],
            "source_git_commit": run_manifest["git_commit"],
            "formal_config_sha256": contract["config_sha256"],
            "adapter_model_sha256": adapter_sha256,
            "gpu_uuid": gpu_uuid,
            "worker": {
                "shard_id": shard.shard_id,
                "start_index": shard.start_index,
                "end_index": shard.end_index,
                "physical_batch_size": physical_batch_size,
                "cuda_device_index": device_index,
            },
        }
        if manifest_path.exists():
            if _read_json(manifest_path) != worker_manifest:
                raise ValueError("formal worker manifest changed")
        else:
            _write_json_exclusive(manifest_path, worker_manifest)
        raw_path = worker_dir / "raw_outputs.jsonl"
        metrics_path = worker_dir / "metrics.json"
        all_records = read_jsonl(contract["data_dir"] / "gsm8k_records.jsonl")
        records = sorted(
            (row for row in all_records if row["protocol_split"] == evaluation["split"]),
            key=lambda row: (row["source_index"], row["record_id"]),
        )
        shard_records = records_for_formal_shard(records, shard)
        completed = read_jsonl(raw_path) if raw_path.exists() else []
        next_offset = validate_formal_worker_prefix(
            rows=completed,
            frozen_shard_records=shard_records,
            shard_id=shard.shard_id,
        )
        if metrics_path.exists():
            if next_offset != shard.count:
                raise ValueError("formal worker metrics exist before shard completion")
            print(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "stage": "formal_eval_worker",
                        "run_id": run_manifest["run_id"],
                        "hashes": {"raw_outputs_sha256": file_sha256(raw_path)},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        set_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        from eg_sft.experiment.phase2_v8_snapshot import frozen_model_source

        model_source, source_kwargs = frozen_model_source(
            contract["protocol"]["model"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_source, **source_kwargs, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            model_source,
            **source_kwargs,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=str(contract["config"]["training"]["attention_implementation"]),
        )
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to(device)
        model.config.use_cache = True
        model.eval()
        torch.cuda.synchronize(device)
        model_load_seconds = time.perf_counter() - load_started
        gsm_test = load_dataset(
            contract["protocol"]["datasets"]["gsm8k"]["repo_id"],
            contract["protocol"]["datasets"]["gsm8k"]["config"],
            split="test",
            revision=contract["protocol"]["datasets"]["gsm8k"]["revision"],
        )
        generation_started = time.perf_counter()
        batches = contiguous_record_batches(
            records=shard_records,
            start_index=next_offset,
            batch_size=physical_batch_size,
        )
        for batch_start, batch_records in batches:
            source_rows = []
            prompts = []
            for record in batch_records:
                source_row = gsm_test[int(record["source_index"])]
                validate_gsm8k_source_row(record, source_row)
                source_rows.append(source_row)
                prompts.append(build_evaluation_prompt(source_row["question"]))
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(evaluation["max_input_length"]),
            ).to(device)
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(evaluation["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            token_rows = generated_token_rows(
                generated_ids=generated,
                padded_input_width=input_width,
            )
            scored_rows = []
            for batch_offset, (record, source_row, token_ids) in enumerate(
                zip(batch_records, source_rows, token_rows, strict=True)
            ):
                raw_output = tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ).strip()
                scored = score_generation(
                    record=record,
                    gold_answer_text=source_row["answer"],
                    generated_text=raw_output,
                )
                offset = batch_start + batch_offset
                scored.update(
                    {
                        "shard_id": shard.shard_id,
                        "shard_offset": offset,
                        "global_record_index": shard.start_index + offset,
                    }
                )
                record_generated_token_ids(
                    scored_row=scored,
                    token_ids=token_ids,
                    identifiable_v4=identifiable_v4,
                    eos_token_id=tokenizer.eos_token_id,
                    canonical_decoded_text=raw_output,
                    parser_input=raw_output,
                )
                scored_rows.append(scored)
            append_jsonl_rows_fsynced(raw_path, scored_rows)
            batch_end = batch_start + len(batch_records)
            if (
                batch_end // 25 > batch_start // 25
                or batch_end == shard.count
            ):
                print(
                    json.dumps(
                        {
                            "status": "RUNNING",
                            "stage": "formal_eval_worker",
                            "progress": f"{batch_end}/{shard.count}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_started
        rows = read_jsonl(raw_path)
        validate_formal_worker_prefix(
            rows=rows,
            frozen_shard_records=shard_records,
            shard_id=shard.shard_id,
        )
        if len(rows) != shard.count:
            raise ValueError("formal worker did not complete its frozen shard")
        metrics = {
            "status": "PASS",
            "record_count": len(rows),
            "model_load_seconds": model_load_seconds,
            "generation_seconds": generation_seconds,
            "worker_wall_seconds": time.perf_counter() - invocation_started,
            "peak_allocated_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_memory_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "gpu_uuid": gpu_uuid,
            "adapter_model_sha256": adapter_sha256,
            "raw_outputs_sha256": file_sha256(raw_path),
            "accuracy_withheld": True,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        if metrics["peak_allocated_memory_gib"] > float(
            contract["config"]["resources"]["max_worker_peak_allocated_gib"]
        ):
            raise RuntimeError("formal evaluation worker exceeded its memory guard")
        _write_json_exclusive(metrics_path, metrics)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "stage": "formal_eval_worker",
                    "run_id": run_manifest["run_id"],
                    "hashes": {
                        "adapter_model_sha256": adapter_sha256,
                        "raw_outputs_sha256": file_sha256(raw_path),
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        if worker_dir is not None:
            _append_jsonl(
                worker_dir / "failures.jsonl",
                {
                    "event": "formal_eval_worker_failure",
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "model_load_seconds_this_invocation": model_load_seconds,
                    "generation_seconds_this_invocation": generation_seconds,
                },
            )
        raise


if __name__ == "__main__":
    main()
