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
from eg_sft.eval.parser import parse_numeric_answer_marker_v2  # noqa: E402
from eg_sft.utils.io import read_jsonl, write_csv, write_jsonl  # noqa: E402

PROMPT_TEMPLATES = {
    "current_completion": "Problem: {prompt}\nFinal numeric answer =",
    "qwen3_answer_only": (
        "Answer the numerical problem with only one final numeric value. "
        "Do not include reasoning, words, LaTeX, units, or a question mark.\n"
        "Problem: {prompt}\nFinal numeric answer:"
    ),
    "qwen3_chat_no_think_answer_only": (
        "Answer with only one final numeric value. Do not include reasoning, words, "
        "LaTeX, units, or a question mark."
    ),
    "qwen3_chat_strict_single_number_v1": (
        "You are being evaluated by an exact parser. Output exactly one numeric value "
        "and nothing else. Do not include reasoning, equations, Markdown, LaTeX, units, "
        "words, punctuation, or a question mark. Do any calculation privately; the final "
        "assistant message must be only the number."
    ),
    "qwen3_chat_cot_final_answer_v1": (
        "Solve the numerical problem carefully. You may write concise calculation steps, "
        "but the final line must be exactly: Final answer: <number>. Do not include units."
    ),
    "qwen3_chat_think_final_answer_v1": (
        "Solve the numerical problem carefully. Use reasoning if needed. The final visible "
        "line must be exactly: Final answer: <number>. Do not include units."
    ),
    "qwen3_chat_exact_arithmetic_final_answer_v1": (
        "Solve the numerical problem by exact decimal arithmetic. Compute every product, "
        "do not round intermediate values, then sum all products. The final line must be "
        "exactly: Final answer: <number>. Do not include units."
    ),
    "qwen3_completion_strict_single_number_v1": (
        "You are being evaluated by an exact parser. Output exactly one numeric value "
        "and nothing else. Do not include reasoning, equations, Markdown, LaTeX, units, "
        "words, punctuation, or a question mark. Do any calculation privately; the final "
        "answer must be only the number.\n"
        "Problem: {prompt}\n"
        "Return exactly one numeric value.\n"
    ),
    "math_cot_final_answer_v1": (
        "Solve the problem. You may write concise calculation steps.\n"
        "End with exactly one separate line in this form: Final answer: <number>\n"
        "Problem: {prompt}\n"
    ),
    "exact_arithmetic_final_answer_v1": (
        "Solve the numerical problem by exact decimal arithmetic.\n"
        "Compute every product, do not round intermediate values, then sum all products.\n"
        "End with exactly one separate line in this form: Final answer: <number>\n"
        "Problem: {prompt}\n"
    ),
}
PROMPT_TEMPLATE = PROMPT_TEMPLATES["current_completion"]
PARSER_VERSION = "parse_numeric_answer_marker_v2_fallback_last_number"
BASELINE_0_5B_PARSER_V2_ACCURACY = 0.28
CHAT_PROMPT_VARIANTS = {
    "qwen3_chat_no_think_answer_only",
    "qwen3_chat_strict_single_number_v1",
    "qwen3_chat_cot_final_answer_v1",
    "qwen3_chat_think_final_answer_v1",
    "qwen3_chat_exact_arithmetic_final_answer_v1",
}


def load_causal_lm_with_dtype_compat(
    model_cls,
    model_name: str,
    *,
    prefer_legacy_torch_dtype: bool = False,
):
    """Load a causal LM across Transformers dtype API compatibility paths.

    Transformers 4.57 accepts ``dtype`` but some Qwen3 configurations raise an
    AttributeError inside that path. Older releases instead reject ``dtype``
    with TypeError. In both cases, the legacy ``torch_dtype`` argument loads the
    same frozen weights and is the compatibility fallback.
    """

    if prefer_legacy_torch_dtype:
        return model_cls.from_pretrained(model_name, torch_dtype="auto")

    try:
        return model_cls.from_pretrained(model_name, dtype="auto")
    except (TypeError, AttributeError):
        return model_cls.from_pretrained(model_name, torch_dtype="auto")


def load_tokenizer_with_local_snapshot_compat(
    auto_tokenizer_cls,
    pretrained_tokenizer_fast_cls,
    model_name: str,
):
    """Load a tokenizer without changing frozen local snapshot assets.

    Transformers 4.57.2 can coerce a local Qwen3 ``config.json`` to a plain
    dictionary inside ``AutoTokenizer``.  For a complete local snapshot, use
    the already-audited direct ``tokenizer.json`` route and preserve the exact
    tokenizer fields that determine prompt serialization.
    """

    try:
        return auto_tokenizer_cls.from_pretrained(model_name), "AutoTokenizer.from_pretrained"
    except AttributeError:
        snapshot = Path(model_name)
        tokenizer_path = snapshot / "tokenizer.json"
        tokenizer_config_path = snapshot / "tokenizer_config.json"
        if not snapshot.is_dir() or not tokenizer_path.is_file() or not tokenizer_config_path.is_file():
            raise

        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        preserved_keys = (
            "chat_template",
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "model_max_length",
            "clean_up_tokenization_spaces",
        )
        kwargs = {
            key: tokenizer_config[key]
            for key in preserved_keys
            if key in tokenizer_config
        }
        tokenizer = pretrained_tokenizer_fast_cls(
            tokenizer_file=str(tokenizer_path),
            **kwargs,
        )
        return tokenizer, "PreTrainedTokenizerFast(local tokenizer.json + frozen fields)"
THINKING_PROMPT_VARIANTS = {"qwen3_chat_think_final_answer_v1"}


def qwen3_enable_thinking_for_variant(prompt_variant: str) -> bool:
    return prompt_variant in THINKING_PROMPT_VARIANTS


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def build_prompt(
    problem_prompt: str,
    prompt_variant: str = "current_completion",
    tokenizer: object | None = None,
) -> str:
    if prompt_variant in CHAT_PROMPT_VARIANTS:
        if tokenizer is None:
            raise ValueError(f"{prompt_variant} requires a tokenizer")
        if prompt_variant == "qwen3_chat_strict_single_number_v1":
            user_content = f"Problem: {problem_prompt}\nReturn exactly one numeric value."
        elif prompt_variant in {
            "qwen3_chat_cot_final_answer_v1",
            "qwen3_chat_think_final_answer_v1",
            "qwen3_chat_exact_arithmetic_final_answer_v1",
        }:
            user_content = (
                f"Problem: {problem_prompt}\n"
                "End with exactly one separate line: Final answer: <number>"
            )
        else:
            user_content = f"Problem: {problem_prompt}"
        messages = [
            {
                "role": "system",
                "content": PROMPT_TEMPLATES[prompt_variant],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        enable_thinking = qwen3_enable_thinking_for_variant(prompt_variant)
        try:
            return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return PROMPT_TEMPLATES[prompt_variant].format(prompt=problem_prompt)


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


def output_dir_for_model(output_root: Path, model: str, run_name: str = "") -> Path:
    return output_root / (run_name or safe_model_name(model))


def raw_outputs_path(output_root: str, model: str, run_name: str = "") -> str:
    return str(
        Path(output_root)
        / (run_name or safe_model_name(model))
        / "scale_model_diagnostic_outputs.jsonl"
    ).replace("\\", "/")


def eos_token_id_list(tokenizer: object) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, (list, tuple, set)):
        return [int(value) for value in eos_token_id if value is not None]
    return [int(eos_token_id)]


def infer_generation_termination(
    generated_token_ids: list[int],
    continuation: str,
    max_new_tokens: int,
    eos_token_ids: list[int],
    stop_at_next_problem: bool,
) -> dict[str, object]:
    eos_token_id_set = set(eos_token_ids)
    ended_by_eos = bool(eos_token_id_set) and any(
        token_id in eos_token_id_set for token_id in generated_token_ids
    )
    stop_at_next_problem_triggered = stop_at_next_problem and continuation.find("\nProblem:") > 0
    generated_token_count = len(generated_token_ids)
    reached_max_new_tokens = (
        generated_token_count >= max_new_tokens
        and not ended_by_eos
        and not stop_at_next_problem_triggered
    )
    if ended_by_eos:
        stopping_reason = "eos"
    elif stop_at_next_problem_triggered:
        stopping_reason = "stop_at_next_problem"
    elif reached_max_new_tokens:
        stopping_reason = "max_new_tokens"
    else:
        stopping_reason = "stopped_before_max_unknown"

    return {
        "generated_token_count": generated_token_count,
        "ended_by_eos": ended_by_eos,
        "reached_max_new_tokens": reached_max_new_tokens,
        "stop_at_next_problem_triggered": stop_at_next_problem_triggered,
        "stopping_reason": stopping_reason,
    }


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
    marker_hits = sum(row["parser_mode"] != "last_number_fallback" for row in diagnostics)
    fallback_hits = sum(row["parser_mode"] == "last_number_fallback" for row in diagnostics)
    accuracy = correct / total if total else 0
    return [
        {
            "condition": metadata.get("condition", "scale_model_diagnostic"),
            "split": "dev_diagnostic",
            "n": total,
            "parse_success_rate": round(parse_success / total, 6) if total else 0,
            "numeric_accuracy": round(accuracy, 6),
            "gain_vs_qwen2_5_0_5b_parser_v2": round(
                accuracy - BASELINE_0_5B_PARSER_V2_ACCURACY,
                6,
            ),
            "answer_marker_rate": round(marker_hits / total, 6) if total else 0,
            "last_number_fallback_rate": round(fallback_hits / total, 6) if total else 0,
            "model": metadata["model"],
            "model_revision": metadata.get("model_revision") or "",
            "tokenizer_revision": metadata.get("tokenizer_revision") or "",
            "dtype": metadata.get("dtype") or "",
            "device": metadata["device"]["type"],
            "seed": metadata["seed"],
            "prompt_variant": metadata.get("prompt_variant", "current_completion"),
            "prompt_template": metadata["prompt_template"],
            "decoding_config": json.dumps(metadata["generation_config"], sort_keys=True),
            "parser_version": metadata["parser_version"],
            "raw_outputs_path": metadata["raw_outputs_path"],
            "note": (
                "Dev-only scale diagnostic. A higher score is evidence about model scale or "
                "model generation, not evidence that error-guided selection or LoRA/SFT works."
            ),
        }
    ]


def write_failure_artifact(output_dir: Path, metadata: dict, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.update({"status": "failed", "failure_reason": reason})
    (output_dir / "scale_model_diagnostic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--input", default="data/samples/dev_diagnostic.jsonl")
    parser.add_argument("--output-root", default="results/model_diagnostic_scale")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--condition", default="scale_model_diagnostic")
    parser.add_argument(
        "--prompt-variant",
        default="current_completion",
        choices=sorted(PROMPT_TEMPLATES),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit; 0 means all rows.")
    parser.add_argument("--run-name", default="", help="Optional output directory name under --output-root.")
    parser.add_argument(
        "--stop-at-next-problem",
        action="store_true",
        help="Stop generation after the continuation begins a new Problem block.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing model output directory.")
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    output_dir = output_dir_for_model(output_root, args.model, args.run_name)
    input_path = ROOT / args.input
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory already exists; pass --overwrite to replace: {output_dir}")

    metadata: dict = {
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_id": args.condition,
        "condition": args.condition,
        "model": args.model,
        "input": args.input,
        "seed": args.seed,
        "prompt_variant": args.prompt_variant,
        "prompt_template": PROMPT_TEMPLATES[args.prompt_variant],
        "parser_version": PARSER_VERSION,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "stop_at_next_problem": args.stop_at_next_problem,
            "termination_metadata_recorded": True,
            "qwen3_enable_thinking": qwen3_enable_thinking_for_variant(args.prompt_variant),
        },
        "baseline_0_5b_parser_v2_accuracy": BASELINE_0_5B_PARSER_V2_ACCURACY,
        "raw_outputs_path": raw_outputs_path(args.output_root, args.model, args.run_name),
        "python": sys.version,
        "platform": platform.platform(),
        "forbidden_claim": (
            "This diagnostic does not show LoRA/SFT or error-guided selection effectiveness."
        ),
    }

    if not input_path.exists():
        raise SystemExit("Missing dev diagnostic data. Run: python scripts/generate_data.py --all")

    try:
        import torch
        import transformers
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            PreTrainedTokenizerFast,
            StoppingCriteria,
            StoppingCriteriaList,
        )
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
        tokenizer, tokenizer_load_method = load_tokenizer_with_local_snapshot_compat(
            AutoTokenizer,
            PreTrainedTokenizerFast,
            args.model,
        )
        model_config = AutoConfig.from_pretrained(args.model)
        model = load_causal_lm_with_dtype_compat(
            AutoModelForCausalLM,
            args.model,
            prefer_legacy_torch_dtype=getattr(model_config, "model_type", "") == "qwen3",
        )
    except Exception as exc:  # pragma: no cover - network/cache dependent
        write_failure_artifact(output_dir, metadata, f"Model load failed: {exc}")
        raise SystemExit(f"Model load failed: {exc}") from exc

    model.to(device)
    model.eval()
    model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision, tokenizer_revision_source = tokenizer_revision_metadata(tokenizer, model_revision)
    metadata["model_revision"] = model_revision
    metadata["tokenizer_load_method"] = tokenizer_load_method
    metadata["tokenizer_call"] = {
        "return_tensors": "pt",
        "return_token_type_ids": False,
    }
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
    eos_token_ids = eos_token_id_list(tokenizer)

    class StopAtNextProblemCriteria(StoppingCriteria):
        def __init__(self, prompt_len: int) -> None:
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[no-untyped-def]
            continuation_ids = input_ids[0][self.prompt_len :]
            if continuation_ids.numel() == 0:
                return False
            continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            marker_pos = continuation.find("\nProblem:")
            return marker_pos > 0

    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            prompt = build_prompt(row["prompt"], args.prompt_variant, tokenizer)
            # Decoder-only Qwen models do not consume token_type_ids.  The
            # generic direct tokenizer loader otherwise emits that field even
            # though input_ids and attention_mask are identical.
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(device)
            input_len = inputs["input_ids"].shape[-1]
            stopping_criteria = None
            if args.stop_at_next_problem:
                stopping_criteria = StoppingCriteriaList([StopAtNextProblemCriteria(input_len)])
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
            generated_ids = output_ids[0][input_len:]
            continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_token_ids = [int(token_id) for token_id in generated_ids.tolist()]
            termination = infer_generation_termination(
                generated_token_ids,
                continuation,
                args.max_new_tokens,
                eos_token_ids,
                args.stop_at_next_problem,
            )
            parsed, parser_mode = parse_numeric_answer_marker_v2(continuation)
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
                    "parser_mode": parser_mode,
                    "parse_success": parsed is not None,
                    "numeric_accuracy": correct,
                    "output_length": len(continuation.split()),
                    "generated_token_count": termination["generated_token_count"],
                    "ended_by_eos": termination["ended_by_eos"],
                    "reached_max_new_tokens": termination["reached_max_new_tokens"],
                    "stop_at_next_problem_triggered": termination["stop_at_next_problem_triggered"],
                    "stopping_reason": termination["stopping_reason"],
                    "error_type": error_type,
                    "model": args.model,
                    "model_revision": metadata.get("model_revision"),
                    "seed": args.seed,
                    "parser_version": PARSER_VERSION,
                    "prompt_variant": args.prompt_variant,
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

    write_jsonl(output_dir / "scale_model_diagnostic_outputs.jsonl", diagnostics)
    write_csv(output_dir / "scale_model_diagnostic_summary.csv", summary_rows(diagnostics, metadata))
    write_csv(output_dir / "scale_model_error_profile.csv", profile)
    write_csv(output_dir / "scale_model_error_profile_by_type.csv", error_type_profile)
    (output_dir / "scale_model_diagnostic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    accuracy = sum(row["numeric_accuracy"] for row in diagnostics) / len(diagnostics)
    print(f"wrote scale diagnostics={len(diagnostics)} accuracy={accuracy:.3f} output_dir={output_dir}")


if __name__ == "__main__":
    main()
