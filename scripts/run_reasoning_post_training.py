"""Run a pinned, traceable arithmetic SFT or continue its adapter with real GRPO."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
import traceback
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.research_upgrade import read_object, source_identity  # noqa: E402
from eg_sft.experiment.run_manifest import create_run_manifest, stable_config_hash  # noqa: E402
from eg_sft.gsm8k.parser import parse_generated_answer  # noqa: E402
from eg_sft.selection.query_groups import load_jsonl  # noqa: E402
from eg_sft.training.lora_audit import audit_lora_parameters  # noqa: E402
from eg_sft.training.overfit import build_tokenized_overfit_examples  # noqa: E402
from eg_sft.training.reasoning import (  # noqa: E402
    RewardTrace, adapter_fingerprint, build_grpo_trainer, build_sft_trainer, format_reward,
    reasoning_partitions, reasoning_rows,
)


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def evaluate(model, tokenizer, rows: list[dict], output: Path, max_new_tokens: int) -> dict:
    import torch
    from decimal import Decimal

    model.eval()
    model.config.use_cache = True
    correct = 0
    parse_failures = 0
    with output.open("x", encoding="utf-8") as handle, torch.no_grad():
        for row in rows:
            inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            parsed = parse_generated_answer(text)
            matched = parsed.ok and parsed.value == Decimal(row["answer"])
            correct += int(matched)
            parse_failures += int(not parsed.ok)
            handle.write(json.dumps({"record_id": row["record_id"], "completion": text,
                                     "gold_answer": row["answer"], "correct": matched,
                                     "parse_status": parsed.status}) + "\n")
    model.config.use_cache = False
    return {"count": len(rows), "correct": correct, "parse_failures": parse_failures,
            "accuracy": correct / len(rows), "decoding": "greedy", "max_new_tokens": max_new_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sft", "grpo"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path)
    args = parser.parse_args()
    config = read_object(args.config)
    if config["schema_version"] != "reasoning-post-training-v1" or config["mode"] != "feasibility":
        raise ValueError("This runner implements feasibility only; freeze a new protocol for research")
    if (args.stage == "grpo") != (args.parent_run is not None):
        raise ValueError("GRPO requires a completed parent SFT run; SFT does not accept one")
    if config["grpo"]["reward_weights"] != [1.0, 0.0]:
        raise ValueError("Frozen reward: correctness only; format is separately logged at zero weight")
    source_path = (ROOT / config["source_config"]).resolve()
    source_path.relative_to(ROOT.resolve())
    source = read_object(source_path)
    records = load_jsonl(args.data_manifest_dir / "gsm8k_records.jsonl")
    partition = config["partition"]
    train_records, validation_records = reasoning_partitions(records, **partition)
    split_identity = {
        "train": [row["record_id"] for row in train_records],
        "validation": [row["record_id"] for row in validation_records],
        "record_manifest_sha256": stable_config_hash({"records": records}),
    }
    parent = None
    if args.parent_run:
        parent = read_object(args.parent_run / "COMPLETE.json")
        if parent["status"] != "PASS" or parent["stage"] != "sft":
            raise ValueError("Parent must be a successfully completed SFT feasibility run")
        if parent["split_identity"] != split_identity or parent["source_config"] != source:
            raise ValueError("Parent model/data identities differ from this GRPO study")
        parent_adapter_config = read_object(args.parent_run / "adapter" / "adapter_config.json")
        if parent_adapter_config["base_model_name_or_path"] != source["model"]["repo_id"]:
            raise ValueError("Parent adapter refers to a different base model")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("AutoDL runner requires a BF16-capable CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Single-GPU runner: expose exactly one CUDA device")
    code_identity = source_identity(ROOT)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "datasets", "accelerate")}
    if args.stage == "grpo":
        versions["trl"] = importlib.metadata.version("trl")
        if versions["trl"] != "0.26.2":
            raise RuntimeError("Use the pinned TRL 0.26.2 environment")
    stage_config = config[args.stage]
    started = time.perf_counter()
    run_dir, manifest = create_run_manifest(
        output_root=args.output_root.resolve(), repo_root=ROOT,
        stage=f"reasoning_{args.stage}_feasibility", config=config, seed=stage_config["seed"],
        command=[sys.executable, *sys.argv],
        dataset_revisions={source["datasets"]["gsm8k"]["repo_id"]:
                           source["datasets"]["gsm8k"]["revision"]},
        model_revision=source["model"]["revision"],
        extra={"gpu_name": torch.cuda.get_device_name(0), "versions": versions,
               "code_identity": code_identity,
               "parent_run_id": parent["run_id"] if parent else None,
               "source_config": source, "split_identity": split_identity},
    )
    try:
        set_seed(stage_config["seed"])
        write_json(run_dir / "splits.json", split_identity)
        dataset_config = source["datasets"]["gsm8k"]
        dataset = load_dataset(dataset_config["repo_id"], dataset_config["config"],
                               revision=dataset_config["revision"], split="train")
        train_rows = reasoning_rows(train_records, dataset)
        validation_rows = reasoning_rows(validation_records, dataset)
        model_config = source["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_config["repo_id"],
                                                   revision=model_config["revision"], use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        for row in train_rows + validation_rows:
            if len(tokenizer(row["prompt"])["input_ids"]) > config["max_prompt_length"]:
                raise ValueError(f"Prompt exceeds frozen length; no silent truncation: {row['record_id']}")
        model = AutoModelForCausalLM.from_pretrained(model_config["repo_id"],
                                                     revision=model_config["revision"],
                                                     dtype=torch.bfloat16).to("cuda")
        model.config.use_cache = False
        if args.stage == "sft":
            model = get_peft_model(model, LoraConfig(**config["lora"], task_type="CAUSAL_LM",
                                                    revision=model_config["revision"]))
        else:
            model = PeftModel.from_pretrained(model, args.parent_run / "adapter", is_trainable=True)
            if adapter_fingerprint(model) != parent["adapter_sha256"]:
                raise ValueError("Loaded parent adapter fingerprint differs from SFT completion")
        write_json(run_dir / "parameter_audit.json", audit_lora_parameters(model).to_dict())
        before = evaluate(model, tokenizer, validation_rows, run_dir / "generations_before.jsonl",
                          config["evaluation_max_new_tokens"])
        initial_fingerprint = adapter_fingerprint(model)
        torch.cuda.reset_peak_memory_stats()
        trace = None
        if args.stage == "sft":
            source_rows = [dataset[int(record["source_index"])] for record in train_records]
            examples, token_audit = build_tokenized_overfit_examples(
                tokenizer=tokenizer, rows=source_rows,
                record_ids=[row["record_id"] for row in train_rows], max_length=config["max_length"])
            write_json(run_dir / "token_audit.json", {"examples": token_audit})
            trainer = build_sft_trainer(model=model, tokenizer=tokenizer, examples=examples,
                                        output_dir=run_dir / "checkpoints", config=stage_config)
        else:
            trace = RewardTrace(run_dir / "rollouts.jsonl", stage_config["num_generations"])
            trainer = build_grpo_trainer(model=model, tokenizer=tokenizer, rows=train_rows,
                                         output_dir=run_dir / "checkpoints", config=stage_config,
                                         reward_funcs=[trace.reward, format_reward])
        training = trainer.train()
        if int(trainer.state.global_step) != stage_config["max_steps"]:
            raise RuntimeError("Training did not complete the configured step count")
        if trace is not None and trace.informative_groups == 0:
            raise RuntimeError("No group contained different correctness rewards; GRPO not qualified")
        final_fingerprint = adapter_fingerprint(trainer.model)
        if final_fingerprint == initial_fingerprint:
            raise RuntimeError("Adapter did not change; cannot report training qualification")
        after = evaluate(trainer.model, tokenizer, validation_rows,
                         run_dir / "generations_after.jsonl", config["evaluation_max_new_tokens"])
        trainer.model.save_pretrained(run_dir / "adapter")
        tokenizer.save_pretrained(run_dir / "tokenizer")
        write_json(run_dir / "training_metrics.json", {"metrics": training.metrics,
                                                       "log_history": trainer.state.log_history})
        complete = {"schema_version": "reasoning-post-training-complete-v1", "status": "PASS",
                    "mode": "feasibility", "stage": args.stage, "run_id": manifest["run_id"],
                    "source_config": source, "split_identity": split_identity,
                    "before": before, "after": after, "adapter_sha256": final_fingerprint,
                    "global_step": trainer.state.global_step,
                    "parent_run_id": parent["run_id"] if parent else None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                    "informative_groups": trace.informative_groups if trace else None,
                    "claim_boundary": config["claim_boundary"]}
        write_json(run_dir / "COMPLETE.json", complete)
        print(json.dumps({"status": "PASS", "run_dir": str(run_dir), "mode": "feasibility"}))
    except Exception as error:
        write_json(run_dir / "FAILED.json", {"status": "FAILED", "exception_type": type(error).__name__,
                                              "traceback": traceback.format_exc(),
                                              "partial_outputs_published": False})
        raise


if __name__ == "__main__":
    main()
