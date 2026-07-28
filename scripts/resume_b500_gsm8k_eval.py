"""Resume a saved B=500 adapter evaluation with local thermal throttling."""

from __future__ import annotations

import argparse
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
from eg_sft.experiment.run_manifest import stable_config_hash  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _gpu_temperature_c() -> int:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = [
        int(line.strip())
        for line in process.stdout.splitlines()
        if line.strip()
    ]
    if len(values) != 1:
        raise RuntimeError(f"expected one GPU temperature, received {values}")
    return values[0]


def _thermal_pause(
    *,
    pause_at_c: int,
    resume_at_c: int,
    poll_seconds: float,
    events_path: Path,
) -> None:
    temperature = _gpu_temperature_c()
    if temperature < pause_at_c:
        return
    started = datetime.now(UTC)
    print(
        f"thermal_pause temperature_c={temperature} "
        f"resume_at_c={resume_at_c}",
        flush=True,
    )
    while temperature > resume_at_c:
        time.sleep(poll_seconds)
        temperature = _gpu_temperature_c()
    event = {
        "started_at_utc": started.isoformat(),
        "resumed_at_utc": datetime.now(UTC).isoformat(),
        "trigger_temperature_c": pause_at_c,
        "resume_temperature_c": temperature,
    }
    with events_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    print(f"thermal_resume temperature_c={temperature}", flush=True)


def _ensure_resume_manifest(
    *,
    path: Path,
    payload: dict[str, Any],
) -> None:
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise ValueError("resume manifest changed after evaluation started")
        return
    _write_json_exclusive(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--recipe-config", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-directory-name",
        default="evaluation_resume_v1",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("resumable evaluation requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support BF16")
    protocol_path = args.protocol_config.resolve()
    recipe_path = args.recipe_config.resolve()
    execution_path = args.execution_config.resolve()
    run_dir = args.run_dir.resolve()
    data_manifest_dir = args.data_manifest_dir.resolve()
    protocol = _read_json(protocol_path)
    recipe = _read_json(recipe_path)
    execution = _read_json(execution_path)
    original_manifest = _read_json(run_dir / "manifest.json")
    evaluation = recipe["evaluation"]
    model_config = protocol["model"]
    gsm_config = protocol["datasets"]["gsm8k"]

    if original_manifest["config"]["protocol_version"] != recipe["protocol_version"]:
        raise ValueError("run protocol and recipe protocol do not match")
    if original_manifest["model_revision"] != model_config["revision"]:
        raise ValueError("run model revision and protocol model do not match")
    if original_manifest["seed"] != recipe["engineering_closure"]["training_seed"]:
        raise ValueError("run seed is not the frozen engineering seed")
    if evaluation["prompt_version"] != PROMPT_VERSION:
        raise ValueError("prompt implementation changed after freezing")
    if int(execution["physical_batch_size"]) != 1:
        raise ValueError("local thermal policy requires physical batch size 1")
    if int(execution["resume_at_temperature_c"]) >= int(
        execution["pause_at_temperature_c"]
    ):
        raise ValueError("resume temperature must be below pause temperature")
    if (
        not args.resume_directory_name
        or Path(args.resume_directory_name).name != args.resume_directory_name
    ):
        raise ValueError("resume directory name must be one safe path segment")

    adapter_path = run_dir / "adapter" / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError("saved adapter weights are missing")
    resume_dir = run_dir / args.resume_directory_name
    resume_dir.mkdir(parents=True, exist_ok=True)
    raw_path = resume_dir / "raw_outputs.jsonl"
    metrics_path = resume_dir / "metrics.json"
    events_path = resume_dir / "thermal_events.jsonl"
    resume_manifest_path = resume_dir / "manifest.json"
    existing_resume_manifest = (
        _read_json(resume_manifest_path)
        if resume_manifest_path.exists()
        else None
    )
    if existing_resume_manifest is None:
        initial_completed_row_count = (
            len(read_jsonl(raw_path)) if raw_path.exists() else 0
        )
        initial_raw_outputs_sha256 = (
            file_sha256(raw_path) if raw_path.exists() else None
        )
    else:
        initial_completed_row_count = int(
            existing_resume_manifest["initial_completed_row_count"]
        )
        initial_raw_outputs_sha256 = existing_resume_manifest[
            "initial_raw_outputs_sha256"
        ]
    resume_manifest = {
        "execution_policy": execution,
        "execution_policy_sha256": file_sha256(execution_path),
        "semantic_evaluation": {
            key: evaluation[key]
            for key in execution["semantic_generation_fields_unchanged"]
        },
        "semantic_evaluation_hash": stable_config_hash(evaluation),
        "source_run_id": original_manifest["run_id"],
        "source_run_git_commit": original_manifest["git_commit"],
        "resume_code_git_commit": _git_commit(),
        "adapter_model_sha256": file_sha256(adapter_path),
        "protocol_config_sha256": file_sha256(protocol_path),
        "recipe_config_sha256": file_sha256(recipe_path),
        "dataset_revisions": original_manifest["dataset_revisions"],
        "model_revision": model_config["revision"],
        "training_seed": original_manifest["seed"],
        "initial_completed_row_count": initial_completed_row_count,
        "initial_raw_outputs_sha256": initial_raw_outputs_sha256,
    }
    _ensure_resume_manifest(
        path=resume_manifest_path,
        payload=resume_manifest,
    )

    records = [
        row
        for row in read_jsonl(data_manifest_dir / "gsm8k_records.jsonl")
        if row["protocol_split"] == "held_out_test"
    ]
    records.sort(key=lambda row: (row["source_index"], row["record_id"]))
    if len(records) != int(evaluation["example_count"]):
        raise ValueError("held-out test count changed")
    completed = read_jsonl(raw_path) if raw_path.exists() else []
    next_index = validate_completed_prefix(
        completed_rows=completed,
        frozen_records=records,
    )
    if metrics_path.exists():
        if next_index != len(records):
            raise ValueError("final metrics exist before all rows are complete")
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "completed": next_index,
                    "metrics": _read_json(metrics_path),
                },
                indent=2,
            )
        )
        return

    set_seed(int(original_manifest["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir / "tokenizer",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, run_dir / "adapter").to("cuda")
    model.config.use_cache = True
    model.eval()
    gsm_test = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        split="test",
        revision=gsm_config["revision"],
    )
    started = time.perf_counter()
    generated_tokens = 0
    peak_memory_bytes = 0
    torch.cuda.reset_peak_memory_stats()

    raw_mode = "a" if raw_path.exists() else "x"
    with raw_path.open(raw_mode, encoding="utf-8", newline="\n") as output:
        for index in range(next_index, len(records)):
            if index % int(execution["temperature_check_every_examples"]) == 0:
                _thermal_pause(
                    pause_at_c=int(execution["pause_at_temperature_c"]),
                    resume_at_c=int(execution["resume_at_temperature_c"]),
                    poll_seconds=float(execution["temperature_poll_seconds"]),
                    events_path=events_path,
                )
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
            generated_tokens += int(
                (token_ids != tokenizer.pad_token_id).sum().item()
            )
            raw_output = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()
            scored = score_generation(
                record=record,
                gold_answer_text=row["answer"],
                generated_text=raw_output,
            )
            output.write(
                json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n"
            )
            output.flush()
            time.sleep(float(execution["inter_example_sleep_seconds"]))
            if (index + 1) % 10 == 0 or index + 1 == len(records):
                print(f"evaluation={index + 1}/{len(records)}", flush=True)
        torch.cuda.synchronize()
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    rows = read_jsonl(raw_path)
    validate_completed_prefix(completed_rows=rows, frozen_records=records)
    elapsed_seconds = time.perf_counter() - started
    metrics = {
        **aggregate_gsm8k_metrics(rows),
        "source_run_id": original_manifest["run_id"],
        "adapter_model_sha256": file_sha256(adapter_path),
        "raw_outputs_sha256": file_sha256(raw_path),
        "generated_tokens_this_invocation": generated_tokens,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gib": peak_memory_bytes / 1024**3,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "This is one random B=500 engineering run. It does not compare "
            "selectors or estimate seed variance."
        ),
    }
    _write_json_exclusive(metrics_path, metrics)
    print(json.dumps({"status": "complete", **metrics}, indent=2))


if __name__ == "__main__":
    main()
