"""Train, save, reload, and evaluate one frozen B=500 LoRA run."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
        raise ValueError("evaluation has zero shifted supervised tokens")
    return weighted_loss / shifted_supervised_tokens, shifted_supervised_tokens


def _held_out_records(path: Path, expected_count: int) -> list[dict[str, Any]]:
    records = [
        row
        for row in read_jsonl(path)
        if row["protocol_split"] == "held_out_test"
    ]
    records.sort(key=lambda row: (row["source_index"], row["record_id"]))
    if len(records) != expected_count:
        raise ValueError("held-out test count does not match frozen recipe")
    return records


@torch.inference_mode()
def _evaluate_gsm8k(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    rows: list[dict[str, str]],
    evaluation: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if evaluation["prompt_version"] != PROMPT_VERSION:
        raise ValueError("frozen prompt version does not match implementation")
    prompts = [build_evaluation_prompt(row["question"]) for row in rows]
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()
    output_rows: list[dict[str, Any]] = []
    generated_tokens = 0
    started = time.perf_counter()
    batch_size = int(evaluation["batch_size"])
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(evaluation["max_input_length"]),
        ).to(device)
        input_width = int(encoded["input_ids"].shape[1])
        generated = model.generate(
            **encoded,
            do_sample=bool(evaluation["do_sample"]),
            num_beams=int(evaluation["num_beams"]),
            max_new_tokens=int(evaluation["max_new_tokens"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuations = generated[:, input_width:]
        for offset, token_ids in enumerate(continuations):
            nonpad_tokens = int(
                (token_ids != tokenizer.pad_token_id).sum().item()
            )
            generated_tokens += nonpad_tokens
            raw_output = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()
            index = start + offset
            output_rows.append(
                score_generation(
                    record=records[index],
                    gold_answer_text=rows[index]["answer"],
                    generated_text=raw_output,
                )
            )
        print(f"evaluation={len(output_rows)}/{len(records)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    correct = sum(bool(row["numeric_correct"]) for row in output_rows)
    parsed = sum(row["parse_status"] == "ok" for row in output_rows)
    strict = sum(row["strict_parse_status"] == "ok" for row in output_rows)
    fallback = sum(
        row["parse_mode"] == "last_numeric_fallback" for row in output_rows
    )
    return output_rows, {
        "example_count": len(output_rows),
        "numeric_correct_count": correct,
        "numeric_accuracy": correct / len(output_rows),
        "parsed_count": parsed,
        "parse_rate": parsed / len(output_rows),
        "strict_parsed_count": strict,
        "strict_parse_rate": strict / len(output_rows),
        "fallback_parsed_count": fallback,
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--recipe-config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strategy", choices=["random", "rds_all", "rds_error"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("B=500 training and evaluation require CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support BF16")
    device = torch.device("cuda")

    protocol = _read_json(args.protocol_config.resolve())
    recipe = _read_json(args.recipe_config.resolve())
    training = recipe["training"]
    evaluation = recipe["evaluation"]
    selection_config = recipe["selection"]
    if args.seed not in recipe["formal_training_seeds"]:
        raise ValueError("seed is not one of the three frozen formal seeds")
    if args.strategy not in selection_config["allowed_strategies"]:
        raise ValueError("strategy is not allowed by the frozen recipe")
    if training["nominal_effective_batch_size"] != (
        training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
    ):
        raise ValueError("effective batch size is inconsistent")
    if evaluation["split"] != "held_out_test":
        raise ValueError("evaluation split must remain held_out_test")

    selection_manifest_path = args.selection_manifest.resolve()
    selection_manifest = _read_json(selection_manifest_path)
    selected = validate_selection_manifest(
        selection_manifest,
        expected_strategy=args.strategy,
        expected_budget=int(selection_config["budget"]),
        expected_selection_seed=int(selection_config["selection_seed"]),
    )
    if selected_id_sha256(selected) != selection_manifest["selected_id_sha256"]:
        raise ValueError("selected candidate ID hash mismatch")

    model_config = protocol["model"]
    gsm_config = protocol["datasets"]["gsm8k"]
    candidate_config = protocol["datasets"]["candidate_pool"]
    run_config = {
        "protocol_version": recipe["protocol_version"],
        "strategy": args.strategy,
        "selection_manifest_sha256": file_sha256(selection_manifest_path),
        "selected_id_sha256": selection_manifest["selected_id_sha256"],
        "model": model_config,
        "datasets": {
            "gsm8k": gsm_config,
            "candidate_pool": candidate_config,
        },
        "training": training,
        "evaluation": evaluation,
    }
    run_dir, manifest = create_run_manifest(
        output_root=args.output_root.resolve(),
        repo_root=ROOT,
        stage=f"b500_lora_{args.strategy}",
        config=run_config,
        seed=args.seed,
        command=[sys.executable, *sys.argv],
        dataset_revisions={
            gsm_config["repo_id"]: gsm_config["revision"],
            candidate_config["repo_id"]: candidate_config["revision"],
        },
        model_revision=model_config["revision"],
        extra={
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        },
    )

    try:
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
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
                raise ValueError(
                    f"token audit changed for {candidate['candidate_id']}"
                )
            train_examples.append(example)
            token_audit.append(audit)

        gsm_train = load_dataset(
            gsm_config["repo_id"],
            gsm_config["config"],
            split="train",
            revision=gsm_config["revision"],
        )
        all_gsm_records = read_jsonl(
            args.data_manifest_dir.resolve() / "gsm8k_records.jsonl"
        )
        development_records = sorted(
            (
                row
                for row in all_gsm_records
                if row["protocol_split"] == "development"
            ),
            key=lambda row: (row["source_index"], row["record_id"]),
        )
        development_rows = [
            gsm_train[int(record["source_index"])]
            for record in development_records
        ]
        for record, row in zip(
            development_records, development_rows, strict=True
        ):
            validate_gsm8k_source_row(record, row)
        development_examples, development_audit = (
            build_tokenized_overfit_examples(
                tokenizer=tokenizer,
                rows=development_rows,
                record_ids=[row["record_id"] for row in development_records],
                max_length=int(training["max_length"]),
            )
        )
        collator = ResponseOnlyCollator(
            pad_token_id=int(tokenizer.pad_token_id)
        )
        development_loader = DataLoader(
            development_examples,
            batch_size=4,
            shuffle=False,
            collate_fn=collator,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model = get_peft_model(
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
        parameter_report = audit_lora_parameters(model)
        pre_validation_loss, validation_tokens = _mean_token_loss(
            model, development_loader, device
        )

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        micro_batch_size = int(training["micro_batch_size"])
        accumulation = int(training["gradient_accumulation_steps"])
        epochs = int(training["epochs"])
        training_loader = DataLoader(
            train_examples,
            batch_size=micro_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            collate_fn=collator,
        )
        total_micro_batches = len(training_loader) * epochs
        optimizer_steps_planned = math.ceil(total_micro_batches / accumulation)
        warmup_steps = math.ceil(
            optimizer_steps_planned * float(training["warmup_ratio"])
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=optimizer_steps_planned,
        )

        optimizer.zero_grad(set_to_none=True)
        training_started = time.perf_counter()
        optimizer_steps = 0
        supervised_tokens_seen = 0
        gradient_audited = False
        epoch_metrics: list[dict[str, Any]] = []
        pending_micro_batches = 0
        for epoch in range(epochs):
            model.train()
            weighted_loss = 0.0
            shifted_tokens = 0
            for batch in training_loader:
                batch = _to_device(batch, device)
                token_count = int(
                    (batch["labels"][..., 1:] != -100).sum().item()
                )
                loss = model(**batch).loss
                (loss / accumulation).backward()
                pending_micro_batches += 1
                if not gradient_audited:
                    audit_lora_gradients(model)
                    gradient_audited = True
                weighted_loss += float(loss.detach().item()) * token_count
                shifted_tokens += token_count
                supervised_tokens_seen += token_count
                if pending_micro_batches == accumulation:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        float(training["gradient_clipping"]),
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    pending_micro_batches = 0
            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "train_token_loss": weighted_loss / shifted_tokens,
                    "shifted_supervised_tokens": shifted_tokens,
                    "optimizer_steps_completed": optimizer_steps,
                }
            )
            print(
                f"epoch={epoch + 1}/{epochs} "
                f"train_token_loss={weighted_loss / shifted_tokens:.6f} "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if pending_micro_batches:
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                float(training["gradient_clipping"]),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        post_validation_loss, _ = _mean_token_loss(
            model, development_loader, device
        )
        peak_training_memory = int(torch.cuda.max_memory_allocated())

        adapter_dir = run_dir / "adapter"
        tokenizer_dir = run_dir / "tokenizer"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(tokenizer_dir)
        pre_reload_adapter_sha256 = file_sha256(
            adapter_dir / "adapter_model.safetensors"
        )

        del scheduler, optimizer, trainable_parameters, model
        gc.collect()
        torch.cuda.empty_cache()

        reloaded_base = AutoModelForCausalLM.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        reloaded_model = PeftModel.from_pretrained(
            reloaded_base,
            adapter_dir,
        ).to(device)
        reloaded_validation_loss, _ = _mean_token_loss(
            reloaded_model, development_loader, device
        )
        reload_loss_difference = abs(
            reloaded_validation_loss - post_validation_loss
        )

        gsm_test = load_dataset(
            gsm_config["repo_id"],
            gsm_config["config"],
            split="test",
            revision=gsm_config["revision"],
        )
        held_out_records = _held_out_records(
            args.data_manifest_dir.resolve() / "gsm8k_records.jsonl",
            int(evaluation["example_count"]),
        )
        held_out_rows = [
            gsm_test[int(record["source_index"])] for record in held_out_records
        ]
        for record, row in zip(held_out_records, held_out_rows, strict=True):
            validate_gsm8k_source_row(record, row)
        raw_outputs, evaluation_metrics = _evaluate_gsm8k(
            model=reloaded_model,
            tokenizer=tokenizer,
            records=held_out_records,
            rows=held_out_rows,
            evaluation=evaluation,
            device=device,
        )
        peak_total_memory = int(torch.cuda.max_memory_allocated())

        _write_jsonl(run_dir / "raw_outputs.jsonl", raw_outputs)
        _write_json(run_dir / "epoch_metrics.json", epoch_metrics)
        _write_json(run_dir / "training_token_audit.json", token_audit)
        _write_json(run_dir / "development_token_audit.json", development_audit)
        metrics = {
            "protocol_version": recipe["protocol_version"],
            "strategy": args.strategy,
            "seed": args.seed,
            "selected_count": len(selected),
            "selected_id_sha256": selection_manifest["selected_id_sha256"],
            "pre_validation_token_loss": pre_validation_loss,
            "post_validation_token_loss": post_validation_loss,
            "reloaded_validation_token_loss": reloaded_validation_loss,
            "adapter_reload_loss_absolute_difference": reload_loss_difference,
            "adapter_reload_gate_difference_at_most_1e_6": (
                reload_loss_difference <= 1e-6
            ),
            "validation_shifted_supervised_tokens": validation_tokens,
            "epochs": epochs,
            "optimizer_steps_planned": optimizer_steps_planned,
            "optimizer_steps_completed": optimizer_steps,
            "warmup_steps": warmup_steps,
            "supervised_tokens_seen": supervised_tokens_seen,
            "training_seconds": training_seconds,
            "supervised_tokens_per_second": (
                supervised_tokens_seen / training_seconds
            ),
            "peak_training_memory_bytes": peak_training_memory,
            "peak_training_memory_gib": peak_training_memory / 1024**3,
            "peak_total_memory_bytes": peak_total_memory,
            "peak_total_memory_gib": peak_total_memory / 1024**3,
            "trainable_parameters": parameter_report.trainable_parameters,
            "total_parameters": parameter_report.total_parameters,
            "trainable_fraction": (
                parameter_report.trainable_parameters
                / parameter_report.total_parameters
            ),
            "adapter_model_sha256": pre_reload_adapter_sha256,
            "evaluation": evaluation_metrics,
            "claim_boundary": (
                "This run establishes the frozen B=500 engineering pipeline. "
                "A single random-seed result does not compare selectors."
            ),
        }
        _write_json(run_dir / "metrics.json", metrics)
        print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))
    except Exception as error:
        failure_path = run_dir / "failure.json"
        if not failure_path.exists():
            _write_json(
                failure_path,
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "manifest": manifest,
                },
            )
        raise


if __name__ == "__main__":
    main()
