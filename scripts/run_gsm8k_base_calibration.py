from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    PROMPT_VERSION,
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402


ALLOWED_SPLITS = {
    "interface_calibration",
    "selection_diagnostic",
    "development",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_protocol_records(
    path: Path,
    *,
    protocol_split: str,
    limit: int,
) -> list[dict[str, Any]]:
    if protocol_split not in ALLOWED_SPLITS:
        raise ValueError(
            f"protocol_split must be one of {sorted(ALLOWED_SPLITS)}; "
            "held-out test is intentionally unavailable"
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["protocol_split"] == protocol_split:
                records.append(record)
    records.sort(key=lambda record: (record["source_index"], record["record_id"]))
    if not 0 < limit <= len(records):
        raise ValueError(
            f"limit must be in [1, {len(records)}] for {protocol_split}"
        )
    return records[:limit]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen base-model prompt on a non-test GSM8K split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-split",
        choices=sorted(ALLOWED_SPLITS),
        default="interface_calibration",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("generation requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support BF16")

    config = _read_json(args.config.resolve())
    model_config = config["model"]
    gsm_config = config["datasets"]["gsm8k"]
    run_config = {
        "model": model_config,
        "gsm8k": gsm_config,
        "protocol_split": args.protocol_split,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "prompt_version": PROMPT_VERSION,
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
        },
        "dtype": "bfloat16",
    }
    run_dir, manifest = create_run_manifest(
        output_root=args.output_root.resolve(),
        repo_root=ROOT,
        stage=f"gsm8k_base_{args.protocol_split}",
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
        records = _read_protocol_records(
            args.data_manifest_dir.resolve() / "gsm8k_records.jsonl",
            protocol_split=args.protocol_split,
            limit=args.limit,
        )
        gsm = load_dataset(
            gsm_config["repo_id"],
            gsm_config["config"],
            split="train",
            revision=gsm_config["revision"],
        )
        rows = [gsm[int(record["source_index"])] for record in records]
        prompts = [build_evaluation_prompt(row["question"]) for row in rows]

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda")
        model.eval()

        output_rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        generated_tokens = 0
        with torch.inference_mode():
            for start in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[start : start + args.batch_size]
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_length,
                ).to("cuda")
                input_width = int(encoded["input_ids"].shape[1])
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
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
                    row_index = start + offset
                    output_rows.append(
                        score_generation(
                            record=records[row_index],
                            gold_answer_text=rows[row_index]["answer"],
                            generated_text=raw_output,
                        )
                    )
                print(
                    f"completed={len(output_rows)}/{len(records)}",
                    flush=True,
                )

        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        strict_parsed_count = sum(
            row["strict_parse_status"] == "ok" for row in output_rows
        )
        parsed_count = sum(row["parse_status"] == "ok" for row in output_rows)
        fallback_parsed_count = sum(
            row["parse_mode"] == "last_numeric_fallback" for row in output_rows
        )
        correct_count = sum(row["numeric_correct"] for row in output_rows)
        status_counts: dict[str, int] = {}
        for row in output_rows:
            status = str(row["parse_status"])
            status_counts[status] = status_counts.get(status, 0) + 1

        metrics = {
            "protocol_split": args.protocol_split,
            "example_count": len(output_rows),
            "strict_parsed_count": strict_parsed_count,
            "strict_parse_rate": strict_parsed_count / len(output_rows),
            "fallback_parsed_count": fallback_parsed_count,
            "parsed_count": parsed_count,
            "parse_rate": parsed_count / len(output_rows),
            "numeric_correct_count": correct_count,
            "numeric_accuracy": correct_count / len(output_rows),
            "parse_status_counts": status_counts,
            "elapsed_seconds": elapsed_seconds,
            "generated_tokens": generated_tokens,
            "generated_tokens_per_second": generated_tokens / elapsed_seconds,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "prompt_version": PROMPT_VERSION,
            "claim_boundary": (
                "Interface-calibration results are for prompt/parser freezing, "
                "not final model evaluation."
            ),
        }
        _write_jsonl(run_dir / "raw_outputs.jsonl", output_rows)
        with (run_dir / "metrics.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
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
