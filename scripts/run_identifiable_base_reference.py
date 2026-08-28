"""Resumable four-task base-model reference for identifiable-budget v4."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import (  # noqa: E402
    _global_job_lock,
    _read_json,
    _require_clean_git_worktree,
)
from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.arithmetic_ood import (  # noqa: E402
    build_ood_prompt,
    score_ood_generation,
)
from eg_sft.evaluation.cloud_v2_batching import (  # noqa: E402
    append_jsonl_rows_fsynced,
    contiguous_record_batches,
)
from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.evaluation.identifiable_batch_backend import (  # noqa: E402
    generated_token_rows,
    validate_resumable_batch_prefix,
)
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    resolve_phase1_contract,
)
from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_ood_runtime import (  # noqa: E402
    OOD_DATASETS,
    resolve_ood_contract,
    validate_source_row,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


TOTAL_RECORDS = 1319 + 300 + 2067 + 155


def _batch_outputs(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[str], list[list[int]]]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(device)
    input_width = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    token_rows = generated_token_rows(
        generated_ids=generated,
        padded_input_width=input_width,
    )
    if len(token_rows) != len(prompts) or len(prompts) > batch_size:
        raise ValueError("base-reference batch cardinality changed")
    texts = [tokenizer.decode(row, skip_special_tokens=True).strip() for row in token_rows]
    return texts, token_rows


def _run_dataset(
    *,
    run_dir: Path,
    dataset: str,
    records: list[dict[str, Any]],
    source: Any,
    prompt: Callable[[dict[str, Any]], str],
    score: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]],
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset_dir = run_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dataset_dir / "raw_outputs.jsonl"
    metrics_path = dataset_dir / "sealed_metrics.json"
    completed = read_jsonl(raw_path) if raw_path.exists() else []
    next_index = validate_resumable_batch_prefix(
        completed_rows=completed,
        frozen_ids=[str(row["record_id"]) for row in records],
    )
    if metrics_path.exists():
        if next_index != len(records):
            raise ValueError(f"{dataset} metrics exist before output completion")
        return _read_json(metrics_path)

    started = time.perf_counter()
    generated_token_count = 0
    for start, batch_records in contiguous_record_batches(
        records=records,
        start_index=next_index,
        batch_size=batch_size,
    ):
        source_rows = [dict(source[int(record["source_index"])]) for record in batch_records]
        prompts = [prompt(row) for row in source_rows]
        texts, token_rows = _batch_outputs(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        output_rows = []
        for offset, (record, source_row, text, token_ids) in enumerate(
            zip(batch_records, source_rows, texts, token_rows, strict=True)
        ):
            row = score(record, source_row, text)
            row.update(
                {
                    "dataset": dataset,
                    "global_record_index": start + offset,
                    "generated_token_ids": token_ids,
                }
            )
            output_rows.append(row)
            generated_token_count += len(token_ids)
        append_jsonl_rows_fsynced(raw_path, output_rows)
        print(json.dumps({"status": "RUNNING", "dataset": dataset, "progress": f"{start + len(batch_records)}/{len(records)}"}, sort_keys=True), flush=True)

    rows = read_jsonl(raw_path)
    validate_resumable_batch_prefix(
        completed_rows=rows,
        frozen_ids=[str(row["record_id"]) for row in records],
    )
    if len(rows) != len(records):
        raise ValueError(f"{dataset} base-reference output is incomplete")
    elapsed = time.perf_counter() - started
    metrics = {
        "status": "PASS",
        "dataset": dataset,
        "record_count": len(rows),
        "raw_outputs_sha256": file_sha256(raw_path),
        "physical_batch_size": batch_size,
        "generation_seconds_this_invocation": elapsed,
        "generated_tokens_this_invocation": generated_token_count,
        "accuracy_withheld": True,
    }
    write_bytes_exclusive_or_verify(metrics_path, canonical_json_bytes(metrics))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/identifiable_budget_v4_matrix.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/identifiable-v4-base-reference"))
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    anchor = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=config_path,
        cell_id="rep1_random_common_mix_train29",
    )
    ood = {
        name: resolve_ood_contract(repo_root=ROOT, matrix_config_path=config_path, dataset=name)
        for name in OOD_DATASETS
    }
    if args.contract_only:
        print(json.dumps({"status": "READY", "stage": "base_reference_contract", "record_count": TOTAL_RECORDS, "config_sha256": anchor["config_sha256"], "gpu_accessed": False}, sort_keys=True))
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("base reference requires a BF16 CUDA GPU")
    _require_clean_git_worktree()

    protocol = anchor["protocol"]
    evaluation = anchor["config"]["evaluation"]
    run_config = {
        "study": "identifiable_budget_v4_base_reference",
        "extension_config_sha256": anchor["config_sha256"],
        "model": protocol["model"],
        "batch_size": args.batch_size,
        "seed": args.seed,
        "record_count": TOTAL_RECORDS,
        "datasets": ["gsm8k", *OOD_DATASETS],
        "prompt_version": evaluation["prompt_version"],
        "parser_policy": evaluation["parser_policy"],
        "accuracy_withheld": True,
    }
    if args.resume_run_dir is None:
        run_dir, manifest = create_run_manifest(
            output_root=args.output_root.resolve(),
            repo_root=ROOT,
            stage="identifiable_budget_v4_base_reference",
            config=run_config,
            seed=args.seed,
            command=[sys.executable, *sys.argv],
            dataset_revisions={
                protocol["datasets"]["gsm8k"]["repo_id"]: protocol["datasets"]["gsm8k"]["revision"],
                **{contract["source"]["repo_id"]: contract["source"]["revision"] for contract in ood.values()},
            },
            model_revision=protocol["model"]["revision"],
        )
    else:
        run_dir = args.resume_run_dir.resolve()
        manifest = _read_json(run_dir / "manifest.json")
        if manifest.get("config") != run_config:
            raise ValueError("base-reference resume contract changed")

    with _global_job_lock(args.output_root.resolve()):
        set_seed(args.seed)
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            protocol["model"]["repo_id"],
            revision=protocol["model"]["revision"],
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            protocol["model"]["repo_id"],
            revision=protocol["model"]["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=str(anchor["config"]["training"]["attention_implementation"]),
        ).to(device)
        base.config.use_cache = True
        base.eval()

        gsm_records = sorted(
            [row for row in read_jsonl(anchor["data_dir"] / "gsm8k_records.jsonl") if row["protocol_split"] == evaluation["split"]],
            key=lambda row: (row["source_index"], row["record_id"]),
        )
        gsm_source = load_dataset(
            protocol["datasets"]["gsm8k"]["repo_id"],
            protocol["datasets"]["gsm8k"]["config"],
            split="test",
            revision=protocol["datasets"]["gsm8k"]["revision"],
        )

        def gsm_score(record: dict, source_row: dict, text: str) -> dict:
            validate_gsm8k_source_row(record, source_row)
            return score_generation(record=record, gold_answer_text=source_row["answer"], generated_text=text)

        metrics = [
            _run_dataset(
                run_dir=run_dir,
                dataset="gsm8k",
                records=gsm_records,
                source=gsm_source,
                prompt=lambda row: build_evaluation_prompt(row["question"]),
                score=gsm_score,
                model=base,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_input_length=int(evaluation["max_input_length"]),
                max_new_tokens=int(evaluation["max_new_tokens"]),
                device=device,
            )
        ]
        for name in OOD_DATASETS:
            contract = ood[name]
            spec = contract["source"]
            source = load_dataset(spec["repo_id"], spec["config"], split=spec["split"], revision=spec["revision"])

            def score_ood(record: dict, source_row: dict, text: str, *, _name=name, _spec=spec) -> dict:
                gold = validate_source_row(record=record, raw_row=source_row, answer_field=str(_spec["answer_field"]))
                return score_ood_generation(record=record, gold_value=gold, generated_text=text)

            metrics.append(
                _run_dataset(
                    run_dir=run_dir,
                    dataset=name,
                    records=contract["records"],
                    source=source,
                    prompt=lambda row, _name=name: build_ood_prompt(_name, row),
                    score=score_ood,
                    model=base,
                    tokenizer=tokenizer,
                    batch_size=args.batch_size,
                    max_input_length=int(evaluation["max_input_length"]),
                    max_new_tokens=int(evaluation["max_new_tokens"]),
                    device=device,
                )
            )

        complete = {
            "status": "PASS",
            "run_id": manifest["run_id"],
            "dataset_count": 4,
            "record_count": sum(int(row["record_count"]) for row in metrics),
            "datasets": metrics,
            "accuracy_withheld": True,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        if complete["record_count"] != TOTAL_RECORDS:
            raise ValueError("base-reference record total changed")
        output = run_dir / "base_reference_complete.json"
        write_bytes_exclusive_or_verify(output, canonical_json_bytes(complete))
        del base
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"status": "COMPLETE", "record_count": TOTAL_RECORDS, "artifact_sha256": file_sha256(output), "accuracy_withheld": True}, sort_keys=True))


if __name__ == "__main__":
    main()
