"""Measure one-step LoRA candidate utility twice for ten frozen candidates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import (  # noqa: E402
    candidate_prompt_text,
    sha256_text,
    validate_gsm8k_source_row,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.experiment.utility import (  # noqa: E402
    icc_absolute_agreement,
    mean_supervised_token_loss,
    pearson_correlation,
    to_device,
)
from eg_sft.selection.query_groups import load_jsonl  # noqa: E402
from eg_sft.training.lora_audit import (  # noqa: E402
    audit_lora_gradients,
    audit_lora_parameters,
)
from eg_sft.training.overfit import (  # noqa: E402
    build_tokenized_overfit_examples,
    gsm8k_training_text,
)
from eg_sft.training.response_only import (  # noqa: E402
    ResponseOnlyCollator,
    tokenize_response_only,
)
from eg_sft.training.tulu import tulu_response_only_parts  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _utility_records(path: Path) -> list[dict[str, Any]]:
    records = [
        record
        for record in load_jsonl(path)
        if record["protocol_split"] == "candidate_utility_validation"
    ]
    records.sort(key=lambda row: (row["source_index"], row["record_id"]))
    if len(records) != 128:
        raise ValueError(f"expected 128 utility records, found {len(records)}")
    return records


def _candidate_lookup(path: Path) -> dict[str, dict[str, Any]]:
    candidates = load_jsonl(path)
    lookup = {row["candidate_id"]: row for row in candidates}
    if len(lookup) != len(candidates):
        raise ValueError("candidate pool has duplicate IDs")
    return lookup


def _validate_candidate_row(
    candidate: dict[str, Any],
    raw_row: dict[str, Any],
) -> list[dict[str, str]]:
    messages = raw_row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{candidate['candidate_id']} has invalid messages")
    if sha256_text(candidate_prompt_text(messages)) != candidate["prompt_sha256"]:
        raise ValueError(f"prompt hash mismatch for {candidate['candidate_id']}")
    if sha256_text(str(messages[-1].get("content", ""))) != candidate["response_sha256"]:
        raise ValueError(f"response hash mismatch for {candidate['candidate_id']}")
    return messages


def _first_batch_loader(
    examples: list[dict[str, list[int]]],
    *,
    collator: ResponseOnlyCollator,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        examples[:batch_size],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )


def _load_base_model(
    *,
    repo_id: str,
    revision: str,
    device: torch.device,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--scoring-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--repeat-seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument(
        "--mode",
        choices=["reliability", "formal", "domain"],
        default="reliability",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("utility measurement requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("utility measurement requires BF16")
    if len(set(args.repeat_seeds)) != len(args.repeat_seeds):
        raise ValueError("repeat seeds must be distinct")
    if args.mode == "reliability" and len(args.repeat_seeds) != 2:
        raise ValueError("reliability mode requires exactly two repeat seeds")
    if args.mode in {"formal", "domain"} and len(args.repeat_seeds) != 1:
        raise ValueError("formal and domain modes require one fixed seed")
    device = torch.device("cuda")

    config = _read_json(args.config.resolve())
    model_config = config["model"]
    gsm_config = config["datasets"]["gsm8k"]
    candidate_config = config["datasets"]["candidate_pool"]
    if args.mode == "reliability":
        measurement_candidates = _read_json(
            args.scoring_run_dir.resolve() / "reliability_candidates.json"
        )
        if len(measurement_candidates) != 10:
            raise ValueError(
                "scoring run must provide exactly ten reliability candidates"
            )
    elif args.mode == "formal":
        measurement_candidates = load_jsonl(
            args.scoring_run_dir.resolve() / "candidate_scores.jsonl"
        )
        if len(measurement_candidates) != 96:
            raise ValueError("formal scoring run must provide exactly 96 candidates")
        if not all(
            row.get("response_only_trainable")
            for row in measurement_candidates
        ):
            raise ValueError("formal candidates must all be response-only trainable")
    else:
        measurement_candidates = load_jsonl(
            args.scoring_run_dir.resolve() / "candidate_scores.jsonl"
        )
        if len(measurement_candidates) != 48:
            raise ValueError("domain scoring run must provide exactly 48 candidates")
        if not all(
            row.get("source_dataset") == "openai/gsm8k"
            and row.get("response_only_trainable")
            for row in measurement_candidates
        ):
            raise ValueError(
                "domain candidates must be trainable openai/gsm8k rows"
            )
    if len(
        {row["candidate_id"] for row in measurement_candidates}
    ) != len(measurement_candidates):
        raise ValueError("measurement candidate IDs are not unique")

    run_config = {
        "mode": args.mode,
        "model": model_config,
        "gsm8k": gsm_config,
        "candidate_pool": (
            {
                "source": "gsm8k_in_domain_candidate_pool",
                "candidate_count": len(measurement_candidates),
            }
            if args.mode == "domain"
            else candidate_config
        ),
        "utility_set_size": 128,
        "candidate_count": len(measurement_candidates),
        "repeat_seeds": args.repeat_seeds,
        "measurement": "one_adamw_step_then_response_loss_reduction",
        "max_length": args.max_length,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "gradient_clipping": 1.0,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "all-linear",
            "bias": "none",
        },
        "reliability_gate": (
            {"icc_a1_at_least": 0.90}
            if args.mode == "reliability"
            else None
        ),
    }
    run_dir, manifest = create_run_manifest(
        output_root=args.output_root.resolve(),
        repo_root=ROOT,
        stage=(
            "h1a_utility_reliability10"
            if args.mode == "reliability"
            else
            "h1a_utility_formal96"
            if args.mode == "formal"
            else "h1a_utility_gsm8k_domain48"
        ),
        config=run_config,
        seed=config["seed"],
        command=[sys.executable, *sys.argv],
        dataset_revisions=(
            {gsm_config["repo_id"]: gsm_config["revision"]}
            if args.mode == "domain"
            else {
                gsm_config["repo_id"]: gsm_config["revision"],
                candidate_config["repo_id"]: candidate_config["revision"],
            }
        ),
        model_revision=model_config["revision"],
        extra={
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "scoring_run_dir": str(args.scoring_run_dir.resolve()),
        },
    )

    try:
        utility_records = _utility_records(
            args.data_manifest_dir.resolve() / "gsm8k_records.jsonl"
        )
        if args.mode == "domain":
            gsm_records = load_jsonl(
                args.data_manifest_dir.resolve() / "gsm8k_records.jsonl"
            )
            candidate_lookup = {
                row["record_id"]: row
                for row in gsm_records
                if row["protocol_split"] == "in_domain_candidate_pool"
            }
            if len(candidate_lookup) != 6705:
                raise ValueError("expected 6705 GSM8K in-domain candidates")
        else:
            candidate_lookup = _candidate_lookup(
                args.data_manifest_dir.resolve() / "tulu_candidate_pool.jsonl"
            )
        gsm = load_dataset(
            gsm_config["repo_id"],
            gsm_config["config"],
            split="train",
            revision=gsm_config["revision"],
        )
        tulu = (
            None
            if args.mode == "domain"
            else load_dataset(
                candidate_config["repo_id"],
                candidate_config["config"],
                split="train",
                revision=candidate_config["revision"],
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has no EOS token")
        collator = ResponseOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))

        utility_rows = [gsm[int(record["source_index"])] for record in utility_records]
        for record, row in zip(utility_records, utility_rows, strict=True):
            validate_gsm8k_source_row(record, row)
        utility_examples, utility_token_audit = build_tokenized_overfit_examples(
            tokenizer=tokenizer,
            rows=utility_rows,
            record_ids=[record["record_id"] for record in utility_records],
            max_length=args.max_length,
        )
        utility_loader = DataLoader(
            utility_examples,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        probe_loader = _first_batch_loader(
            utility_examples,
            collator=collator,
            batch_size=args.eval_batch_size,
        )

        candidate_examples: dict[str, dict[str, list[int]]] = {}
        candidate_token_audit: list[dict[str, Any]] = []
        for score_row in measurement_candidates:
            candidate_id = score_row["candidate_id"]
            candidate = candidate_lookup[candidate_id]
            if args.mode == "domain":
                raw_candidate = gsm[int(candidate["source_index"])]
                validate_gsm8k_source_row(candidate, raw_candidate)
                prompt, response = gsm8k_training_text(
                    raw_candidate["question"],
                    raw_candidate["answer"],
                )
            else:
                if tulu is None:
                    raise AssertionError("Tulu dataset was not loaded")
                messages = _validate_candidate_row(
                    candidate,
                    tulu[int(candidate["source_index"])],
                )
                prompt, response = tulu_response_only_parts(
                    messages,
                    eos_token=tokenizer.eos_token,
                )
            tokenized = tokenize_response_only(
                tokenizer,
                prompt=prompt,
                response=response,
                max_length=args.max_length,
                add_eos=True,
            )
            candidate_examples[candidate_id] = tokenized
            candidate_token_audit.append(
                {
                    "candidate_id": candidate_id,
                    "source_index": candidate["source_index"],
                    "total_tokens": len(tokenized["input_ids"]),
                    "supervised_tokens": sum(
                        label != -100 for label in tokenized["labels"]
                    ),
                    "reached_max_length": (
                        len(tokenized["input_ids"]) == args.max_length
                    ),
                }
            )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_model = _load_base_model(
            repo_id=model_config["repo_id"],
            revision=model_config["revision"],
            device=device,
        )

        base_utility_loss = mean_supervised_token_loss(
            base_model,
            utility_loader,
            device,
        )
        base_probe_loss = mean_supervised_token_loss(
            base_model,
            probe_loader,
            device,
        )
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        started = time.perf_counter()
        measurements: list[dict[str, Any]] = []
        measurement_path = run_dir / "utility_measurements.jsonl"
        gradient_audited = False
        with measurement_path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as measurement_file:
            total_measurements = (
                len(measurement_candidates) * len(args.repeat_seeds)
            )
            for candidate_position, score_row in enumerate(
                measurement_candidates
            ):
                candidate_id = score_row["candidate_id"]
                candidate_loader = DataLoader(
                    [candidate_examples[candidate_id]],
                    batch_size=1,
                    shuffle=False,
                    collate_fn=collator,
                )
                for repeat_index, repeat_seed in enumerate(args.repeat_seeds):
                    set_seed(repeat_seed)
                    fresh_base_model = _load_base_model(
                        repo_id=model_config["repo_id"],
                        revision=model_config["revision"],
                        device=device,
                    )
                    model = get_peft_model(
                        fresh_base_model,
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
                    initial_probe_loss = mean_supervised_token_loss(
                        model,
                        probe_loader,
                        device,
                    )
                    initial_probe_difference = abs(
                        initial_probe_loss - base_probe_loss
                    )
                    if initial_probe_difference > 1e-7:
                        raise RuntimeError(
                            "zero-initialized adapter changed probe loss by "
                            f"{initial_probe_difference}"
                        )

                    trainable = [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    optimizer = torch.optim.AdamW(
                        trainable,
                        lr=args.learning_rate,
                        weight_decay=0.0,
                    )
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    candidate_batch = to_device(next(iter(candidate_loader)), device)
                    candidate_train_loss = model(**candidate_batch).loss
                    candidate_train_loss.backward()
                    if not gradient_audited:
                        audit_lora_gradients(model)
                        gradient_audited = True
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    post_utility_loss = mean_supervised_token_loss(
                        model,
                        utility_loader,
                        device,
                    )
                    utility = base_utility_loss - post_utility_loss
                    measurement = {
                        "candidate_position": candidate_position,
                        "candidate_id": candidate_id,
                        "repeat_index": repeat_index,
                        "repeat_seed": repeat_seed,
                        "all_query_rank": score_row["all_query_rank"],
                        "all_query_score": score_row["all_query_score"],
                        "error_query_rank": score_row["error_query_rank"],
                        "error_query_score": score_row["error_query_score"],
                        "base_utility_loss": base_utility_loss,
                        "post_utility_loss": post_utility_loss,
                        "utility": utility,
                        "candidate_train_loss": float(candidate_train_loss.item()),
                        "gradient_norm_before_clipping": gradient_norm,
                        "initial_probe_loss_difference": initial_probe_difference,
                        "trainable_parameters": (
                            parameter_report.trainable_parameters
                        ),
                    }
                    measurements.append(measurement)
                    measurement_file.write(
                        json.dumps(
                            measurement,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    measurement_file.flush()
                    print(
                        f"completed={len(measurements)}/{total_measurements} "
                        f"candidate={candidate_id} seed={repeat_seed} "
                        f"utility={utility:.10f}",
                        flush=True,
                    )

                    del optimizer, trainable, candidate_batch, model, fresh_base_model
                    gc.collect()
                    torch.cuda.empty_cache()

        elapsed = time.perf_counter() - started
        by_candidate: dict[str, dict[int, float]] = defaultdict(dict)
        for row in measurements:
            by_candidate[row["candidate_id"]][row["repeat_seed"]] = row["utility"]
        measurement_matrix = [
            [by_candidate[row["candidate_id"]][seed] for seed in args.repeat_seeds]
            for row in measurement_candidates
        ]
        peak_memory = int(torch.cuda.max_memory_allocated())
        metrics = {
            "mode": args.mode,
            "candidate_count": len(measurement_candidates),
            "repeat_count": len(args.repeat_seeds),
            "measurement_count": len(measurements),
            "utility_set_size": 128,
            "base_utility_loss": base_utility_loss,
            "repeat_means": [
                sum(row[repeat_index] for row in measurement_matrix)
                / len(measurement_matrix)
                for repeat_index in range(len(args.repeat_seeds))
            ],
            "positive_utility_measurements": sum(
                row["utility"] > 0 for row in measurements
            ),
            "utility_min": min(row["utility"] for row in measurements),
            "utility_max": max(row["utility"] for row in measurements),
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_memory,
            "peak_memory_gib": peak_memory / 1024**3,
            "claim_boundary": (
                "This ten-candidate run tests utility measurement reliability. "
                "It is not the preregistered 96-candidate H1a effect test."
                if args.mode == "reliability"
                else
                "This run measures formal candidate utility. H1a requires "
                "the preregistered partial correlation and permutation analysis."
                if args.mode == "formal"
                else
                "This run measures 48 GSM8K in-domain candidate utilities. "
                "The domain boundary conclusion requires frozen H1a statistics."
            ),
        }
        if args.mode == "reliability":
            first_repeat = [row[0] for row in measurement_matrix]
            second_repeat = [row[1] for row in measurement_matrix]
            icc = icc_absolute_agreement(measurement_matrix)
            pearson = pearson_correlation(first_repeat, second_repeat)
            absolute_differences = [
                abs(left - right)
                for left, right in zip(
                    first_repeat,
                    second_repeat,
                    strict=True,
                )
            ]
            metrics.update(
                {
                    "icc_absolute_agreement_a1": icc,
                    "pearson_between_repeats": pearson,
                    "mean_absolute_repeat_difference": (
                        sum(absolute_differences)
                        / len(absolute_differences)
                    ),
                    "max_absolute_repeat_difference": max(
                        absolute_differences
                    ),
                    "reliability_gate_icc_at_least_0_90": icc >= 0.90,
                }
            )
            if not math.isfinite(icc) or not math.isfinite(pearson):
                raise RuntimeError("non-finite reliability statistic")
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(
            run_dir / "token_audit.json",
            {
                "utility_examples": utility_token_audit,
                "candidates": candidate_token_audit,
            },
        )
        print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))
    except Exception as error:
        _write_json(
            run_dir / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "manifest": manifest,
            },
        )
        raise


if __name__ == "__main__":
    main()
