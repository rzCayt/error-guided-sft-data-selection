from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.eval.error_taxonomy import classify_error  # noqa: E402
from eg_sft.eval.metrics import numeric_equal  # noqa: E402
from eg_sft.eval.parser import parse_numeric  # noqa: E402
from eg_sft.utils.io import read_jsonl, write_csv, write_jsonl  # noqa: E402

PROMPT_TEMPLATE = "Problem: {prompt}\nFinal numeric answer ="
PARSER_VERSION = "parse_numeric_last_number_v1"


def build_prompt(problem_prompt: str) -> str:
    return PROMPT_TEMPLATE.format(prompt=problem_prompt)


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


def profile_rows(diagnostics: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(lambda: {"count": 0, "failures": 0})
    grouped_error_type = defaultdict(lambda: {"count": 0, "failures": 0})
    for row in diagnostics:
        key = (
            row["task_family"],
            row["difficulty_bucket"],
            row["answer_magnitude_bucket"],
            row["reasoning_length_bucket"],
        )
        grouped[key]["count"] += 1
        grouped[key]["failures"] += int(not row["numeric_accuracy"])

        error_key = key + (row["error_type"],)
        grouped_error_type[error_key]["count"] += 1
        grouped_error_type[error_key]["failures"] += int(not row["numeric_accuracy"])

    profile = []
    for key, stats in sorted(grouped.items()):
        count = stats["count"]
        failures = stats["failures"]
        profile.append(
            {
                "task_family": key[0],
                "difficulty_bucket": key[1],
                "answer_magnitude_bucket": key[2],
                "reasoning_length_bucket": key[3],
                "count": count,
                "failures": failures,
                "error_rate": round(failures / count, 6) if count else 0,
            }
        )

    error_type_profile = []
    for key, stats in sorted(grouped_error_type.items()):
        count = stats["count"]
        failures = stats["failures"]
        error_type_profile.append(
            {
                "task_family": key[0],
                "difficulty_bucket": key[1],
                "answer_magnitude_bucket": key[2],
                "reasoning_length_bucket": key[3],
                "error_type": key[4],
                "count": count,
                "failures": failures,
                "error_rate": round(failures / count, 6) if count else 0,
            }
        )
    return profile, error_type_profile


def summary_rows(diagnostics: list[dict], metadata: dict) -> list[dict]:
    total = len(diagnostics)
    parse_success = sum(bool(row["parse_success"]) for row in diagnostics)
    correct = sum(bool(row["numeric_accuracy"]) for row in diagnostics)
    return [
        {
            "condition": "base_real_diagnostic",
            "split": "dev_diagnostic",
            "n": total,
            "parse_success_rate": round(parse_success / total, 6) if total else 0,
            "numeric_accuracy": round(correct / total, 6) if total else 0,
            "model": metadata["model"],
            "model_revision": metadata.get("model_revision") or "",
            "tokenizer_revision": metadata.get("tokenizer_revision") or "",
            "dtype": metadata.get("dtype") or "",
            "device": metadata["device"]["type"],
            "seed": metadata["seed"],
            "prompt_template": metadata["prompt_template"],
            "decoding_config": json.dumps(metadata["generation_config"], sort_keys=True),
            "parser_version": metadata["parser_version"],
            "note": (
                "Real base-model diagnostic on dev_diagnostic only. This is not a LoRA "
                "comparison and does not show Targeted selection beats Random."
            ),
        }
    ]


def write_failure_artifact(output_dir: Path, metadata: dict, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.update({"status": "failed", "failure_reason": reason})
    (output_dir / "real_base_diagnostic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--input", default="data/samples/dev_diagnostic.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit; 0 means all rows.")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    input_path = ROOT / args.input
    metadata: dict = {
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "input": args.input,
        "seed": args.seed,
        "prompt_template": PROMPT_TEMPLATE,
        "parser_version": PARSER_VERSION,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "python": sys.version,
        "platform": platform.platform(),
    }

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
    if tokenizer_revision_source == "model_config_commit_hash_same_hf_repo":
        metadata["tokenizer_revision_note"] = (
            "AutoTokenizer did not expose a separate commit hash; tokenizer and model were loaded "
            "from the same Hugging Face repo argument, so the model config commit is recorded for "
            "the tokenizer artifact."
        )
    metadata["dtype"] = str(getattr(model, "dtype", ""))

    rows = read_jsonl(input_path)
    if args.limit:
        rows = rows[: args.limit]
        metadata["limit"] = args.limit
    if not rows:
        raise SystemExit("No diagnostic rows to run.")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    diagnostics = []
    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            prompt = build_prompt(row["prompt"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
            parsed = parse_numeric(continuation)
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
                    "task_family": row["task_family"],
                    "difficulty_bucket": buckets["difficulty_bucket"],
                    "answer_magnitude_bucket": buckets["answer_magnitude_bucket"],
                    "reasoning_length_bucket": buckets["reasoning_length_bucket"],
                    "answer": row["answer"],
                    "prompt": prompt,
                    "raw_continuation": continuation,
                    "full_text": full_text,
                    "parsed_prediction": "" if parsed is None else parsed,
                    "parse_success": parsed is not None,
                    "numeric_accuracy": correct,
                    "output_length": len(continuation.split()),
                    "error_type": error_type,
                    "model": args.model,
                    "model_revision": metadata.get("model_revision"),
                    "seed": args.seed,
                    "parser_version": PARSER_VERSION,
                }
            )
            if idx % 10 == 0 or idx == len(rows):
                print(f"generated {idx}/{len(rows)}")

    profile, error_type_profile = profile_rows(diagnostics)
    metadata["status"] = "ok"
    metadata["n"] = len(diagnostics)
    if device == "cuda":
        metadata["cuda_memory"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        }

    write_jsonl(output_dir / "real_base_diagnostic_outputs.jsonl", diagnostics)
    write_csv(output_dir / "real_base_diagnostic_summary.csv", summary_rows(diagnostics, metadata))
    write_csv(output_dir / "real_error_profile.csv", profile)
    write_csv(output_dir / "real_error_profile_by_type.csv", error_type_profile)
    (output_dir / "real_base_diagnostic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    accuracy = sum(row["numeric_accuracy"] for row in diagnostics) / len(diagnostics)
    print(f"wrote real diagnostics={len(diagnostics)} accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
