from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.eval.error_taxonomy import classify_error  # noqa: E402
from eg_sft.eval.metrics import numeric_equal  # noqa: E402
from eg_sft.eval.parser import (  # noqa: E402
    NUMBER_RE,
    parse_numeric_final_answer_or_last_number_with_mode,
)
from eg_sft.utils.io import read_jsonl, write_csv, write_jsonl  # noqa: E402

PARSER_VERSION = "parse_numeric_final_answer_v1_fallback_last_number"
BASELINE_DEV_ACCURACY = 0.21

PROMPT_VARIANTS = {
    "current_direct": "Problem: {prompt}\nFinal numeric answer =",
    "step_by_step_final_answer": (
        "Problem: {prompt}\n"
        "Show the calculation briefly. Put the final result on the last line as: "
        "Final answer: <number>"
    ),
    "final_answer_only": (
        "Problem: {prompt}\n"
        "Return only one line in this exact format: Final answer: <number>"
    ),
}


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def build_prompt(problem_prompt: str, variant_id: str) -> str:
    if variant_id not in PROMPT_VARIANTS:
        raise KeyError(f"Unknown prompt variant: {variant_id}")
    return PROMPT_VARIANTS[variant_id].format(prompt=problem_prompt)


def parse_variant_list(value: str) -> list[str]:
    if value == "all":
        return list(PROMPT_VARIANTS)
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(PROMPT_VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown prompt variants: {', '.join(unknown)}")
    return variants


def should_use_chat_template(model_name: str, tokenizer: object, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(getattr(tokenizer, "chat_template", None)) and "instruct" in model_name.lower()


def render_model_input(
    tokenizer: object,
    model_name: str,
    prompt: str,
    chat_template_mode: str,
) -> tuple[str, str]:
    if should_use_chat_template(model_name, tokenizer, chat_template_mode):
        messages = [
            {
                "role": "system",
                "content": "You are a careful numerical reasoning assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered, "chat_template"
    return prompt, "plain_completion"


def tokenizer_revision_metadata(tokenizer: object, model_revision: str | None) -> tuple[str | None, str]:
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None)
    if tokenizer_revision:
        return tokenizer_revision, "tokenizer._commit_hash"

    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    tokenizer_revision = init_kwargs.get("_commit_hash")
    if tokenizer_revision:
        return tokenizer_revision, "tokenizer.init_kwargs._commit_hash"

    if model_revision:
        return model_revision, "model_config_commit_hash_same_hf_repo"

    return None, "unavailable"


def profile_rows(diagnostics: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: {"count": 0, "failures": 0})
    for row in diagnostics:
        key = (
            row["model"],
            row["prompt_variant"],
            row["task_family"],
            row["difficulty_bucket"],
            row["answer_magnitude_bucket"],
            row["reasoning_length_bucket"],
        )
        grouped[key]["count"] += 1
        grouped[key]["failures"] += int(not row["numeric_accuracy"])

    profile = []
    for key, stats in sorted(grouped.items()):
        count = stats["count"]
        failures = stats["failures"]
        profile.append(
            {
                "model": key[0],
                "prompt_variant": key[1],
                "task_family": key[2],
                "difficulty_bucket": key[3],
                "answer_magnitude_bucket": key[4],
                "reasoning_length_bucket": key[5],
                "count": count,
                "failures": failures,
                "error_rate": round(failures / count, 6) if count else 0,
            }
        )
    return profile


def summary_rows(diagnostics: list[dict], metadata: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in diagnostics:
        grouped[row["prompt_variant"]].append(row)

    summaries = []
    for variant_id, rows in sorted(grouped.items()):
        total = len(rows)
        parse_success = sum(bool(row["parse_success"]) for row in rows)
        correct = sum(bool(row["numeric_accuracy"]) for row in rows)
        marker_count = sum(row["parser_mode"] == "final_answer_marker" for row in rows)
        fallback_count = sum(row["parser_mode"] == "last_number_fallback" for row in rows)
        multi_number_count = sum(int(row["numeric_token_count"]) >= 2 for row in rows)
        accuracy = correct / total if total else 0.0
        summaries.append(
            {
                "condition": "qwen_0_5b_prompt_rescue_gate",
                "split": "dev_diagnostic",
                "model": metadata["model"],
                "prompt_variant": variant_id,
                "n": total,
                "parse_success_rate": round(parse_success / total, 6) if total else 0,
                "numeric_accuracy": round(accuracy, 6),
                "final_answer_marker_rate": round(marker_count / total, 6) if total else 0,
                "last_number_fallback_rate": round(fallback_count / total, 6) if total else 0,
                "multi_number_output_rate": round(multi_number_count / total, 6) if total else 0,
                "absolute_gain_vs_recorded_base": round(accuracy - BASELINE_DEV_ACCURACY, 6),
                "recorded_base_accuracy": BASELINE_DEV_ACCURACY,
                "raw_outputs_path": str(metadata["raw_outputs_path"]),
                "model_revision": metadata.get("model_revision") or "",
                "tokenizer_revision": metadata.get("tokenizer_revision") or "",
                "dtype": metadata.get("dtype") or "",
                "device": metadata["device"]["type"],
                "seed": metadata["seed"],
                "prompt_template": PROMPT_VARIANTS[variant_id],
                "prompt_rendering": metadata["prompt_rendering"],
                "decoding_config": json.dumps(metadata["generation_config"], sort_keys=True),
                "parser_version": metadata["parser_version"],
                "note": (
                    "Dev-only prompt/parser rescue diagnostic. This is not a LoRA result and "
                    "does not show error-guided selection beats random baselines."
                ),
            }
        )
    return summaries


def write_failure_artifact(output_dir: Path, metadata: dict, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.update({"status": "failed", "failure_reason": reason})
    (output_dir / "prompt_rescue_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--input", default="data/samples/dev_diagnostic.jsonl")
    parser.add_argument("--output-dir", default="results/prompt_rescue")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit; 0 means all rows.")
    parser.add_argument("--prompt-variants", default="all", help="all or comma-separated variant ids.")
    parser.add_argument(
        "--chat-template",
        choices=["auto", "always", "never"],
        default="auto",
        help="Use tokenizer chat template for instruct-style models.",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_dir = ROOT / args.output_dir / model_slug(args.model)
    variants = parse_variant_list(args.prompt_variants)
    metadata: dict = {
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "input": args.input,
        "seed": args.seed,
        "prompt_variants": variants,
        "prompt_templates": {variant: PROMPT_VARIANTS[variant] for variant in variants},
        "parser_version": PARSER_VERSION,
        "recorded_base_accuracy": BASELINE_DEV_ACCURACY,
        "chat_template_mode": args.chat_template,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    metadata["raw_outputs_path"] = str(
        Path(args.output_dir) / model_slug(args.model) / "prompt_rescue_outputs.jsonl"
    )

    if not input_path.exists():
        raise SystemExit("Missing dev diagnostic data. Run: python scripts/generate_data.py --all")

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        write_failure_artifact(output_dir, metadata, f"Required inference dependencies unavailable: {exc}")
        raise SystemExit(f"Required inference dependencies unavailable: {exc}") from exc

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata["packages"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    metadata["device"] = {
        "type": device,
        "name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
        except TypeError:  # pragma: no cover - older transformers compatibility
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    except Exception as exc:  # pragma: no cover - network/cache dependent
        write_failure_artifact(output_dir, metadata, f"Model load failed: {exc}")
        raise SystemExit(f"Model load failed: {exc}") from exc

    model.to(device)
    model.eval()
    model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision, tokenizer_revision_source = tokenizer_revision_metadata(tokenizer, model_revision)
    metadata["model_revision"] = model_revision
    metadata["tokenizer_revision"] = tokenizer_revision
    metadata["tokenizer_revision_source"] = tokenizer_revision_source
    metadata["dtype"] = str(getattr(model, "dtype", ""))

    rows = read_jsonl(input_path)
    if args.limit:
        rows = rows[: args.limit]
        metadata["limit"] = args.limit
    if not rows:
        raise SystemExit("No diagnostic rows to run.")

    first_prompt = build_prompt(rows[0]["prompt"], variants[0])
    _, prompt_rendering = render_model_input(tokenizer, args.model, first_prompt, args.chat_template)
    metadata["prompt_rendering"] = prompt_rendering

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    diagnostics = []
    total_generations = len(rows) * len(variants)
    completed = 0
    with torch.no_grad():
        for row in rows:
            for variant_id in variants:
                prompt = build_prompt(row["prompt"], variant_id)
                model_input, prompt_rendering = render_model_input(
                    tokenizer,
                    args.model,
                    prompt,
                    args.chat_template,
                )
                inputs = tokenizer(model_input, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[-1]
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated_ids = output_ids[0][input_len:]
                continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
                full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                parsed, parser_mode = parse_numeric_final_answer_or_last_number_with_mode(continuation)
                correct = numeric_equal(parsed, float(row["answer"]))
                error_type = classify_error(
                    row["task_family"],
                    float(row["answer"]),
                    parsed,
                    continuation,
                )
                buckets = row["buckets"]
                diagnostics.append(
                    {
                        "id": row["id"],
                        "split": row["split"],
                        "model": args.model,
                        "prompt_variant": variant_id,
                        "task_family": row["task_family"],
                        "difficulty_bucket": buckets["difficulty_bucket"],
                        "answer_magnitude_bucket": buckets["answer_magnitude_bucket"],
                        "reasoning_length_bucket": buckets["reasoning_length_bucket"],
                        "answer": row["answer"],
                        "problem_prompt": row["prompt"],
                        "rendered_prompt": model_input,
                        "prompt_rendering": prompt_rendering,
                        "raw_continuation": continuation,
                        "full_text": full_text,
                        "parsed_prediction": "" if parsed is None else parsed,
                        "parser_mode": parser_mode,
                        "numeric_token_count": len(NUMBER_RE.findall(continuation.replace(",", ""))),
                        "parse_success": parsed is not None,
                        "numeric_accuracy": correct,
                        "output_length": len(continuation.split()),
                        "error_type": error_type,
                        "model_revision": metadata.get("model_revision"),
                        "seed": args.seed,
                        "parser_version": PARSER_VERSION,
                    }
                )
                completed += 1
                if completed % 10 == 0 or completed == total_generations:
                    print(f"generated {completed}/{total_generations}")

    metadata["status"] = "ok"
    metadata["n_examples"] = len(rows)
    metadata["n_generations"] = len(diagnostics)
    if device == "cuda":
        metadata["cuda_memory"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        }

    write_jsonl(output_dir / "prompt_rescue_outputs.jsonl", diagnostics)
    write_csv(output_dir / "prompt_rescue_summary.csv", summary_rows(diagnostics, metadata))
    write_csv(output_dir / "prompt_rescue_error_profile.csv", profile_rows(diagnostics))
    (output_dir / "prompt_rescue_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    best = max(summary_rows(diagnostics, metadata), key=lambda row: row["numeric_accuracy"])
    print(
        "wrote prompt rescue diagnostics="
        f"{len(diagnostics)} best_variant={best['prompt_variant']} "
        f"accuracy={best['numeric_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
