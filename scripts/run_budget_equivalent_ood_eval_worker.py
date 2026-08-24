"""Run one resumable arithmetic-OOD evaluation shard for a frozen Phase 1 adapter."""

from __future__ import annotations

import argparse
import gc
import json
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

from eg_sft.evaluation.arithmetic_ood import (  # noqa: E402
    build_ood_prompt,
    score_ood_generation,
)
from eg_sft.evaluation.cloud_v2_batching import append_jsonl_rows_fsynced  # noqa: E402
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    resolve_phase1_contract,
)
from eg_sft.experiment.budget_equivalent_ood_runtime import (  # noqa: E402
    OOD_DATASETS,
    contiguous_shard,
    resolve_ood_contract,
    validate_source_row,
    validate_worker_prefix,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _contract_payload(config_path: Path, dataset: str) -> dict[str, object]:
    contract = resolve_ood_contract(
        repo_root=ROOT,
        matrix_config_path=config_path,
        dataset=dataset,
    )
    return {
        "status": "READY",
        "stage": "budget_equivalent_ood_eval_contract",
        "dataset": dataset,
        "record_count": len(contract["records"]),
        "matrix_config_sha256": contract["matrix_config_sha256"],
        "ood_manifest_sha256": contract["manifest_sha256"],
        "ood_records_sha256": contract["records_sha256"],
        "source_revision": contract["source"]["revision"],
        "gpu_accessed": False,
        "accuracy_withheld": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset", choices=OOD_DATASETS, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    if args.contract_only:
        print(json.dumps(_contract_payload(config_path, args.dataset), sort_keys=True))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required unless --contract-only is used")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("OOD evaluation worker requires a BF16 CUDA GPU")

    _require_clean_git_worktree()
    started = time.perf_counter()
    worker_dir: Path | None = None
    model_load_seconds = 0.0
    generation_seconds = 0.0
    try:
        run_dir = args.run_dir.resolve()
        run_manifest = _read_json(run_dir / "manifest.json")
        cell_id = str(run_manifest["config"]["cell_id"])
        phase1 = resolve_phase1_contract(
            repo_root=ROOT,
            config_path=config_path,
            cell_id=cell_id,
        )
        run_dir.relative_to(phase1["output_root"])
        if run_manifest["config"].get("phase1_config_sha256") != phase1["config_sha256"]:
            raise ValueError("run manifest matrix hash changed")
        ood = resolve_ood_contract(
            repo_root=ROOT,
            matrix_config_path=config_path,
            dataset=args.dataset,
        )
        start_index, end_index, shard_records = contiguous_shard(
            ood["records"],
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )

        training_dir = run_dir / "training_complete"
        training_metrics = _read_json(training_dir / "training_metrics.json")
        if training_metrics.get("status") != "PASS":
            raise ValueError("OOD evaluation requires passed training artifacts")
        adapter_dir = training_dir / "adapter"
        adapter_path = adapter_dir / "adapter_model.safetensors"
        adapter_sha256 = file_sha256(adapter_path)
        if adapter_sha256 != training_metrics.get("adapter_model_sha256"):
            raise ValueError("formal adapter SHA-256 changed")

        device_index = int(phase1["config"]["evaluation"]["cuda_device_index"])
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        gpu_uuid = _gpu_uuid(device_index)
        worker_name = f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
        worker_dir = run_dir / "evaluation" / "ood" / args.dataset / "workers" / worker_name
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_manifest = {
            "worker_schema_version": "budget-equivalent-ood-eval-worker-v1",
            "source_run_id": run_manifest["run_id"],
            "source_git_commit": run_manifest["git_commit"],
            "cell_id": cell_id,
            "matrix_config_sha256": phase1["config_sha256"],
            "ood_manifest_sha256": ood["manifest_sha256"],
            "ood_records_sha256": ood["records_sha256"],
            "adapter_model_sha256": adapter_sha256,
            "gpu_uuid": gpu_uuid,
            "worker": {
                "dataset": args.dataset,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "start_index": start_index,
                "end_index": end_index,
                "physical_batch_size": 1,
                "cuda_device_index": device_index,
            },
        }
        manifest_path = worker_dir / "manifest.json"
        if manifest_path.exists():
            if _read_json(manifest_path) != worker_manifest:
                raise ValueError("OOD worker manifest changed")
        else:
            _write_json_exclusive(manifest_path, worker_manifest)

        raw_path = worker_dir / "raw_outputs.jsonl"
        metrics_path = worker_dir / "metrics.json"
        completed = read_jsonl(raw_path) if raw_path.exists() else []
        next_offset = validate_worker_prefix(
            rows=completed,
            frozen_records=shard_records,
        )
        if metrics_path.exists():
            if next_offset != len(shard_records):
                raise ValueError("OOD worker metrics exist before shard completion")
            print(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "stage": "budget_equivalent_ood_eval_worker",
                        "dataset": args.dataset,
                        "progress": f"{next_offset}/{len(shard_records)}",
                        "accuracy_withheld": True,
                        "raw_outputs_sha256": file_sha256(raw_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

        set_seed(int(phase1["seed"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(training_dir / "tokenizer", use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            phase1["protocol"]["model"]["repo_id"],
            revision=phase1["protocol"]["model"]["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=str(phase1["config"]["training"]["attention_implementation"]),
        )
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to(device)
        model.config.use_cache = True
        model.eval()
        torch.cuda.synchronize(device)
        model_load_seconds = time.perf_counter() - load_started

        spec = ood["source"]
        source = load_dataset(
            spec["repo_id"],
            spec["config"],
            split=spec["split"],
            revision=spec["revision"],
        )
        if len(source) != int(spec["source_count"]):
            raise ValueError("OOD source dataset count changed")
        evaluation = phase1["config"]["evaluation"]
        generation_started = time.perf_counter()
        for offset in range(next_offset, len(shard_records)):
            record = shard_records[offset]
            source_row = dict(source[int(record["source_index"])])
            gold_value = validate_source_row(
                record=record,
                raw_row=source_row,
                answer_field=str(spec["answer_field"]),
            )
            prompt = build_ood_prompt(args.dataset, source_row)
            encoded = tokenizer(
                [prompt],
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
            raw_output = tokenizer.decode(
                generated[0, input_width:], skip_special_tokens=True
            ).strip()
            scored = score_ood_generation(
                record=record,
                gold_value=gold_value,
                generated_text=raw_output,
            )
            scored.update(
                {
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "shard_offset": offset,
                    "global_record_index": start_index + offset,
                }
            )
            append_jsonl_rows_fsynced(raw_path, [scored])
            if (offset + 1) % 25 == 0 or offset + 1 == len(shard_records):
                print(
                    json.dumps(
                        {
                            "status": "RUNNING",
                            "stage": "budget_equivalent_ood_eval_worker",
                            "dataset": args.dataset,
                            "progress": f"{offset + 1}/{len(shard_records)}",
                            "accuracy_withheld": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_started
        rows = read_jsonl(raw_path)
        validate_worker_prefix(rows=rows, frozen_records=shard_records)
        if len(rows) != len(shard_records):
            raise ValueError("OOD worker did not complete its frozen shard")
        metrics = {
            "status": "PASS",
            "dataset": args.dataset,
            "record_count": len(rows),
            "model_load_seconds": model_load_seconds,
            "generation_seconds": generation_seconds,
            "worker_wall_seconds": time.perf_counter() - started,
            "peak_allocated_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_memory_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "gpu_uuid": gpu_uuid,
            "adapter_model_sha256": adapter_sha256,
            "raw_outputs_sha256": file_sha256(raw_path),
            "accuracy_withheld": True,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        if metrics["peak_allocated_memory_gib"] > float(
            phase1["config"]["resources"]["max_worker_peak_allocated_gib"]
        ):
            raise RuntimeError("OOD evaluation worker exceeded its memory guard")
        _write_json_exclusive(metrics_path, metrics)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "stage": "budget_equivalent_ood_eval_worker",
                    "dataset": args.dataset,
                    "progress": f"{len(rows)}/{len(shard_records)}",
                    "accuracy_withheld": True,
                    "raw_outputs_sha256": file_sha256(raw_path),
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
                    "event": "budget_equivalent_ood_worker_failure",
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
