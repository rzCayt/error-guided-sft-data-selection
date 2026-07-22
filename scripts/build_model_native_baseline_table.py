from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.eval.metrics import numeric_equal  # noqa: E402
from eg_sft.eval.parser import NUMBER_RE, parse_numeric_final_marker_only_v4  # noqa: E402
from eg_sft.utils.io import read_jsonl, write_csv, write_jsonl  # noqa: E402

TABLE_VERSION = "model_native_baseline_table_v1"
BOXED_RE = re.compile(r"\\boxed\s*\{?\s*([-+]?(?:\d*\.\d+|\d+))\s*\}?")
FINAL_MARKER_RE = re.compile(
    r"final\s+(?:numeric\s+)?answer\s*(?:is|=|:)|final\s+value\s*(?:is|=|:)",
    flags=re.IGNORECASE,
)


BASELINE_SPECS = [
    {
        "run": "qwen2_5_1_5b_exact_completion_max192",
        "model": "Qwen/Qwen2.5-1.5B",
        "baseline_role": "weak_same_family_reference",
        "interface": "completion",
        "prompt_variant": "exact_arithmetic_final_answer_v1",
        "max_new_tokens": 192,
        "source_outputs_path": "results/strong_baseline_protocol_v2_ab/qwen2_5_1_5b_exact_completion_max192/scale_model_diagnostic_outputs.jsonl",
        "score_policy": "strict_main_only",
        "baseline_status": "not_strong_numeric_failure",
    },
    {
        "run": "qwen3_1_7b_exact_chat_max192",
        "model": "Qwen/Qwen3-1.7B",
        "baseline_role": "small_general_model_native_reference",
        "interface": "chat_native_no_thinking",
        "prompt_variant": "qwen3_chat_exact_arithmetic_final_answer_v1",
        "max_new_tokens": 192,
        "source_outputs_path": "results/strong_baseline_protocol_v2_ab/qwen3_1_7b_exact_chat_max192/scale_model_diagnostic_outputs.jsonl",
        "score_policy": "strict_main",
        "baseline_status": "model_native_reference",
    },
    {
        "run": "qwen3_4b_exact_chat_max192",
        "model": "Qwen/Qwen3-4B",
        "baseline_role": "strongest_small_general_reference",
        "interface": "chat_native_no_thinking",
        "prompt_variant": "qwen3_chat_exact_arithmetic_final_answer_v1",
        "max_new_tokens": 192,
        "source_outputs_path": "results/qwen3_4b_reasoning_protocol_diagnostic/qwen3_4b_exact_arithmetic_final_answer_max192/scale_model_diagnostic_outputs.jsonl",
        "score_policy": "strict_main",
        "baseline_status": "current_strongest_small_general_reference",
    },
    {
        "run": "qwen2_5_math_1_5b_exact_completion_max384",
        "model": "Qwen/Qwen2.5-Math-1.5B",
        "baseline_role": "math_native_auxiliary_sanity_control",
        "interface": "completion_math_native_outputs",
        "prompt_variant": "exact_arithmetic_final_answer_v1",
        "max_new_tokens": 384,
        "source_outputs_path": "results/qwen2_5_math_budget_interface_diagnostic/qwen2_5_math_1_5b_exact_completion_max384/scale_model_diagnostic_outputs.jsonl",
        "score_policy": "math_auxiliary_not_main",
        "baseline_status": "auxiliary_only_boxed_or_formula_signal",
    },
]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def numbers_in_text(text: str) -> list[float]:
    return [float(value) for value in NUMBER_RE.findall(text.replace(",", ""))]


def boxed_numbers(text: str) -> list[float]:
    return [float(value) for value in BOXED_RE.findall(text.replace(",", ""))]


def has_formula_output(text: str) -> bool:
    return (
        "\\times" in text
        or "+" in text
        or "=" in text
        or bool(re.search(r"\d\s*\*\s*\d", text))
    )


def has_final_marker(text: str) -> bool:
    return bool(FINAL_MARKER_RE.search(text))


def row_strict_correct(row: dict) -> bool:
    prediction, _mode = parse_numeric_final_marker_only_v4(str(row.get("raw_continuation", "")))
    return numeric_equal(prediction, float(row["answer"]))


def row_boxed_correct(row: dict) -> bool:
    answer = float(row["answer"])
    return any(numeric_equal(value, answer) for value in boxed_numbers(str(row.get("raw_continuation", ""))))


def row_math_native_aux_correct(row: dict) -> bool:
    raw = str(row.get("raw_continuation", ""))
    answer = float(row["answer"])
    return row_boxed_correct(row) or (
        has_formula_output(raw) and any(numeric_equal(value, answer) for value in numbers_in_text(raw))
    )


def summarize_spec(spec: dict) -> dict:
    path = ROOT / spec["source_outputs_path"]
    if not path.exists():
        raise SystemExit(f"Missing baseline outputs: {path}")

    rows = read_jsonl(path)
    total = len(rows)
    if not rows:
        raise SystemExit(f"No rows in baseline outputs: {path}")

    strict_correct = sum(row_strict_correct(row) for row in rows)
    boxed_correct = sum(row_boxed_correct(row) for row in rows)
    math_aux_correct = sum(row_math_native_aux_correct(row) for row in rows)
    runner_correct = sum(bool(row.get("numeric_accuracy")) for row in rows)
    final_marker_present = sum(has_final_marker(str(row.get("raw_continuation", ""))) for row in rows)
    eos = sum(row.get("stopping_reason") == "eos" for row in rows)
    max_token = sum(row.get("stopping_reason") == "max_new_tokens" for row in rows)

    return {
        "table_version": TABLE_VERSION,
        "run": spec["run"],
        "model": spec["model"],
        "baseline_role": spec["baseline_role"],
        "interface": spec["interface"],
        "prompt_variant": spec["prompt_variant"],
        "max_new_tokens": spec["max_new_tokens"],
        "n": total,
        "main_strict_final_answer_accuracy": round(strict_correct / total, 6),
        "math_native_auxiliary_score": round(math_aux_correct / total, 6)
        if spec["score_policy"] == "math_auxiliary_not_main"
        else "",
        "boxed_correct_rate": round(boxed_correct / total, 6)
        if spec["score_policy"] == "math_auxiliary_not_main"
        else "",
        "runner_fallback_proxy_accuracy": round(runner_correct / total, 6),
        "final_marker_present_rate": round(final_marker_present / total, 6),
        "eos_rate": round(eos / total, 6),
        "max_token_rate": round(max_token / total, 6),
        "score_policy": spec["score_policy"],
        "baseline_status": spec["baseline_status"],
        "source_outputs_path": display_path(path),
        "claim_boundary": (
            "Do not combine strict, math-native auxiliary, and fallback proxy scores. "
            "This table is dev-only and does not show LoRA/SFT/selection effectiveness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/model_native_baseline_table")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    rows = [summarize_spec(spec) for spec in BASELINE_SPECS]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_native_baseline_table.csv"
    jsonl_path = output_dir / "model_native_baseline_table.jsonl"
    metadata_path = output_dir / "model_native_baseline_table_metadata.json"
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    metadata = {
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "table_version": TABLE_VERSION,
        "n_baselines": len(rows),
        "outputs": {
            "csv": display_path(csv_path),
            "jsonl": display_path(jsonl_path),
            "metadata": display_path(metadata_path),
        },
        "policy": {
            "qwen3": "Use chat-native no-thinking final-answer protocol.",
            "qwen2_5_math": "Report math-native auxiliary signal separately; do not add it to strict score.",
            "fallback_proxy": "Runner fallback accuracy is diagnostic only.",
            "forbidden_claim": "No LoRA/SFT/Targeted-vs-Random/selection effectiveness claim.",
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote model-native baseline table rows={len(rows)} path={csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
