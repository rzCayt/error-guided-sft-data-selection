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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.training.lora_audit import (  # noqa: E402
    audit_lora_gradients,
    audit_lora_parameters,
)
from eg_sft.training.overfit import build_tokenized_overfit_examples  # noqa: E402
from eg_sft.training.response_only import ResponseOnlyCollator  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_development_records(path: Path, count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["protocol_split"] == "development":
                records.append(record)
    records.sort(key=lambda record: (record["source_index"], record["record_id"]))
    if len(records) < count:
        raise ValueError(f"only {len(records)} development records for count={count}")
    return records[:count]


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.no_grad()
def _mean_token_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    for batch in loader:
        batch = _to_device(batch, device)
        token_count = int((batch["labels"] != -100).sum().item())
        loss = model(**batch).loss
        weighted_loss += float(loss.item()) * token_count
        supervised_tokens += token_count
    if supervised_tokens == 0:
        raise ValueError("evaluation has zero supervised tokens")
    return weighted_loss / supervised_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed 16-example Qwen LoRA overfit check."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this overfit command requires CUDA")
    device = torch.device("cuda")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support BF16")

    config = _read_json(args.config.resolve())
    model_config = config["model"]
    gsm_config = config["datasets"]["gsm8k"]
    run_config = {
        "model": model_config,
        "gsm8k": gsm_config,
        "examples": args.examples,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "all-linear",
        },
        "dtype": "bfloat16",
    }
    run_dir, manifest = create_run_manifest(
        output_root=args.output_root.resolve(),
        repo_root=ROOT,
        stage="gsm8k_lora_overfit16",
        config=run_config,
        seed=args.seed,
        command=[sys.executable, *sys.argv],
        dataset_revisions={gsm_config["repo_id"]: gsm_config["revision"]},
        model_revision=model_config["revision"],
        extra={
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        },
    )

    try:
        set_seed(args.seed)
        records = _read_development_records(
            args.data_manifest_dir.resolve() / "gsm8k_records.jsonl",
            args.examples,
        )
        gsm = load_dataset(
            gsm_config["repo_id"],
            gsm_config["config"],
            split="train",
            revision=gsm_config["revision"],
        )
        rows = [gsm[int(record["source_index"])] for record in records]
        record_ids = [str(record["record_id"]) for record in records]

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        examples, token_audit = build_tokenized_overfit_examples(
            tokenizer=tokenizer,
            rows=rows,
            record_ids=record_ids,
            max_length=args.max_length,
        )
        collator = ResponseOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))
        evaluation_loader = DataLoader(
            examples,
            batch_size=1,
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
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                bias="none",
            ),
        )
        model.to(device)
        parameter_report = audit_lora_parameters(model)

        pre_loss = _mean_token_loss(model, evaluation_loader, device)
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        training_loader = DataLoader(
            examples,
            batch_size=1,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            collate_fn=collator,
        )

        epoch_metrics: list[dict[str, float | int]] = []
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        total_supervised_tokens = 0
        optimizer_steps = 0
        gradient_audited = False

        for epoch in range(args.epochs):
            model.train()
            epoch_weighted_loss = 0.0
            epoch_tokens = 0
            for batch_index, batch in enumerate(training_loader):
                batch = _to_device(batch, device)
                token_count = int((batch["labels"] != -100).sum().item())
                loss = model(**batch).loss
                (loss / args.gradient_accumulation).backward()

                if not gradient_audited:
                    audit_lora_gradients(model)
                    gradient_audited = True

                epoch_weighted_loss += float(loss.detach().item()) * token_count
                epoch_tokens += token_count
                total_supervised_tokens += token_count

                should_step = (
                    (batch_index + 1) % args.gradient_accumulation == 0
                    or batch_index + 1 == len(training_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

            epoch_loss = epoch_weighted_loss / epoch_tokens
            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "train_token_loss": epoch_loss,
                    "supervised_tokens": epoch_tokens,
                }
            )
            print(
                f"epoch={epoch + 1} train_token_loss={epoch_loss:.6f} "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )

        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        post_loss = _mean_token_loss(model, evaluation_loader, device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())

        adapter_dir = run_dir / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(run_dir / "tokenizer")

        del optimizer, trainable_parameters, model
        gc.collect()
        torch.cuda.empty_cache()

        reloaded_base = AutoModelForCausalLM.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        reloaded_model = PeftModel.from_pretrained(reloaded_base, adapter_dir)
        reloaded_model.to(device)
        reloaded_post_loss = _mean_token_loss(
            reloaded_model, evaluation_loader, device
        )
        reload_loss_difference = abs(reloaded_post_loss - post_loss)

        metrics = {
            "pre_train_token_loss": pre_loss,
            "post_train_token_loss": post_loss,
            "reloaded_post_train_token_loss": reloaded_post_loss,
            "adapter_reload_loss_absolute_difference": reload_loss_difference,
            "loss_ratio_post_over_pre": post_loss / pre_loss,
            "loss_reduction_fraction": 1.0 - post_loss / pre_loss,
            "perplexity_pre": math.exp(min(pre_loss, 20.0)),
            "perplexity_post": math.exp(min(post_loss, 20.0)),
            "epochs": args.epochs,
            "optimizer_steps": optimizer_steps,
            "total_supervised_tokens_seen": total_supervised_tokens,
            "elapsed_seconds": elapsed_seconds,
            "supervised_tokens_per_second": total_supervised_tokens / elapsed_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_gib": peak_memory_bytes / 1024**3,
            "trainable_parameters": parameter_report.trainable_parameters,
            "total_parameters": parameter_report.total_parameters,
            "trainable_fraction": (
                parameter_report.trainable_parameters
                / parameter_report.total_parameters
            ),
            "overfit_gate_loss_ratio_at_most_0_5": post_loss / pre_loss <= 0.5,
            "adapter_reload_gate_difference_at_most_1e_6": (
                reload_loss_difference <= 1e-6
            ),
        }
        with (run_dir / "metrics.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        with (run_dir / "epoch_metrics.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                epoch_metrics, handle, ensure_ascii=False, indent=2, sort_keys=True
            )
            handle.write("\n")
        with (run_dir / "selected_records.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                token_audit, handle, ensure_ascii=False, indent=2, sort_keys=True
            )
            handle.write("\n")

        print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))
    except Exception as error:
        with (run_dir / "failure.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "manifest": manifest,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        raise


if __name__ == "__main__":
    main()
