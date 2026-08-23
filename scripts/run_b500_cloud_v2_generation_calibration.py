"""Run one isolated 128-example cloud-v2 batched-generation calibration."""

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
    _git_commit,
    _gpu_sample,
    _read_json,
    _require_clean_git_worktree,
    _write_json_exclusive,
)

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.cloud_v2_batching import (  # noqa: E402
    append_jsonl_rows_fsynced,
    contiguous_record_batches,
)
from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.evaluation.resumable import (  # noqa: E402
    aggregate_gsm8k_metrics,
    validate_completed_prefix,
)
from eg_sft.experiment.cloud_v2_calibration import (  # noqa: E402
    calibration_run_config,
    read_json_object,
    repository_path,
    resolve_frozen_artifact,
    validate_calibration_config,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _validate_data_bindings(config: dict) -> Path:
    data = config["data_manifest"]
    directory = repository_path(ROOT, str(data["directory"]), label="data manifest")
    for filename, expected in data["required_files"].items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"frozen data artifact changed: {filename}")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("configs/b500_cloud_v2_calibration_v1.json"),
    )
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 4, 8, 16], required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("cloud-v2 generation calibration requires a BF16 CUDA GPU")
    _require_clean_git_worktree()
    calibration_path = args.calibration_config.resolve()
    config = read_json_object(calibration_path)
    validate_calibration_config(config)
    if args.batch_size not in config["generation_batch_sizes"]:
        raise ValueError("batch size is outside the frozen calibration grid")
    protocol_path = resolve_frozen_artifact(
        repo_root=ROOT,
        binding=config["protocol_config"],
        label="protocol config",
    )
    protocol = _read_json(protocol_path)
    data_manifest_dir = _validate_data_bindings(config)
    training_output_root = repository_path(
        ROOT,
        str(config["training_output_root"]),
        label="training calibration output root",
    )
    training_run_dir = args.training_run_dir.resolve()
    training_run_dir.relative_to(training_output_root)
    training_manifest = _read_json(training_run_dir / "manifest.json")
    training_metrics_path = training_run_dir / "training_complete" / "calibration_metrics.json"
    training_metrics = _read_json(training_metrics_path)
    if training_metrics.get("status") != "PASS":
        raise ValueError("generation calibration requires a passed training calibration")
    if training_manifest["config"].get("study_role") != (
        "engineering_calibration_only_excluded_from_formal_matrix"
    ):
        raise ValueError("training adapter is not an isolated calibration artifact")
    adapter_dir = training_run_dir / "training_complete" / "adapter"
    tokenizer_dir = training_run_dir / "training_complete" / "tokenizer"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    adapter_sha256 = file_sha256(adapter_path)
    run_config = {
        **calibration_run_config(
            payload=config,
            profile=None,
            generation_batch_size=args.batch_size,
            adapter_sha256=adapter_sha256,
        ),
        "calibration_config_file_sha256": file_sha256(calibration_path),
        "protocol_config_sha256": file_sha256(protocol_path),
        "source_training_run_id": training_manifest["run_id"],
        "source_training_config_hash": training_manifest["config_hash"],
        "generation_protocol_split": config["generation_protocol_split"],
        "generation_example_count": int(config["generation_example_count"]),
    }
    output_root = repository_path(
        ROOT,
        str(config["generation_output_root"]),
        label="generation calibration output root",
    )
    seed = int(config["training_seed"])
    if args.resume_run_dir is None:
        run_dir, manifest = create_run_manifest(
            output_root=output_root,
            repo_root=ROOT,
            stage=f"cloud_v2_generation_calibration_b{args.batch_size}",
            config=run_config,
            seed=seed,
            command=[str(Path(__file__)), "--batch-size", str(args.batch_size)],
            dataset_revisions={
                protocol["datasets"]["gsm8k"]["repo_id"]: protocol["datasets"]["gsm8k"][
                    "revision"
                ]
            },
            model_revision=protocol["model"]["revision"],
            extra={"gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__},
        )
    else:
        run_dir = args.resume_run_dir.resolve()
        run_dir.relative_to(output_root)
        manifest = _read_json(run_dir / "manifest.json")
        if manifest["config"] != run_config or manifest["seed"] != seed:
            raise ValueError("resume run differs from the frozen generation contract")
        if manifest["git_commit"] != _git_commit():
            raise ValueError("resume must use the generation run's original commit")

    raw_path = run_dir / "raw_outputs.jsonl"
    metrics_path = run_dir / "metrics.json"
    all_records = read_jsonl(data_manifest_dir / "gsm8k_records.jsonl")
    records = sorted(
        (
            row
            for row in all_records
            if row["protocol_split"] == config["generation_protocol_split"]
        ),
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    if len(records) != int(config["generation_example_count"]):
        raise ValueError("development calibration record count changed")
    completed = read_jsonl(raw_path) if raw_path.exists() else []
    next_index = validate_completed_prefix(
        completed_rows=completed,
        frozen_records=records,
    )
    if metrics_path.exists():
        if next_index != len(records):
            raise ValueError("metrics exist before generation calibration is complete")
        print(metrics_path.read_text(encoding="utf-8"))
        return

    guards = config["resource_guards"]
    start_gpu = _gpu_sample()
    if start_gpu["temperature_c"] > float(guards["start_max_temperature_c"]):
        raise RuntimeError("GPU is above the frozen calibration start temperature")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(
        protocol["model"]["repo_id"],
        revision=protocol["model"]["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=str(config["attention_implementation"]),
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to("cuda")
    model.config.use_cache = True
    model.eval()
    gsm_train = load_dataset(
        protocol["datasets"]["gsm8k"]["repo_id"],
        protocol["datasets"]["gsm8k"]["config"],
        split="train",
        revision=protocol["datasets"]["gsm8k"]["revision"],
    )
    started = time.perf_counter()
    generated_token_count = 0
    last_temperature_check = next_index
    batches = contiguous_record_batches(
        records=records,
        start_index=next_index,
        batch_size=args.batch_size,
    )
    for batch_start, batch_records in batches:
        source_rows = [gsm_train[int(record["source_index"])] for record in batch_records]
        for record, source_row in zip(batch_records, source_rows, strict=True):
            validate_gsm8k_source_row(record, source_row)
        prompts = [build_evaluation_prompt(row["question"]) for row in source_rows]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        input_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        scored_rows = []
        for offset, (record, source_row) in enumerate(
            zip(batch_records, source_rows, strict=True)
        ):
            token_ids = generated[offset, input_width:]
            generated_token_count += int(token_ids.numel())
            raw_output = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            scored_rows.append(
                score_generation(
                    record=record,
                    gold_answer_text=source_row["answer"],
                    generated_text=raw_output,
                )
            )
        append_jsonl_rows_fsynced(raw_path, scored_rows)
        completed_count = batch_start + len(batch_records)
        if completed_count - last_temperature_check >= 64 or completed_count == len(records):
            sample = _gpu_sample()
            last_temperature_check = completed_count
            if sample["temperature_c"] >= float(guards["hard_stop_temperature_c"]):
                raise RuntimeError("generation calibration reached its hard temperature stop")
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        if peak_gib > float(guards["max_peak_gpu_memory_gib"]):
            raise RuntimeError("generation calibration exceeded its peak GPU memory guard")
        print(f"generation_calibration={completed_count}/{len(records)}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rows = read_jsonl(raw_path)
    validate_completed_prefix(completed_rows=rows, frozen_records=records)
    metrics = {
        **aggregate_gsm8k_metrics(rows),
        "status": "PASS",
        "study_role": "engineering_calibration_only_excluded_from_formal_matrix",
        "physical_batch_size": args.batch_size,
        "source_training_run_id": training_manifest["run_id"],
        "adapter_model_sha256": adapter_sha256,
        "raw_outputs_sha256": file_sha256(raw_path),
        "generation_seconds_this_invocation": elapsed,
        "generated_token_count_this_invocation": generated_token_count,
        "generated_tokens_per_second": generated_token_count / elapsed,
        "peak_evaluation_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_evaluation_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": "Batching calibration on development only; no held-out result.",
    }
    _write_json_exclusive(metrics_path, metrics)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
