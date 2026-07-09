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

from eg_sft.eval.metrics import numeric_equal  # noqa: E402
from eg_sft.eval.parser import NUMBER_RE, parse_numeric_answer_marker_v2  # noqa: E402
from eg_sft.utils.io import read_jsonl, write_csv, write_jsonl  # noqa: E402

PARSER_VERSION = "parse_answer_marker_v2_final_answer_final_value_fallback_last_number"


def discover_output_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/prompt_rescue_outputs.jsonl"))


def rescore_row(row: dict) -> dict:
    raw = str(row.get("raw_continuation", ""))
    v2_prediction, v2_mode = parse_numeric_answer_marker_v2(raw)
    answer = float(row["answer"])
    v1_correct = bool(row.get("numeric_accuracy"))
    v2_correct = numeric_equal(v2_prediction, answer)
    if v1_correct and not v2_correct:
        flip = "correct_to_incorrect"
    elif not v1_correct and v2_correct:
        flip = "incorrect_to_correct"
    elif v1_correct and v2_correct:
        flip = "correct_stable"
    else:
        flip = "incorrect_stable"

    rescored = dict(row)
    rescored.update(
        {
            "parser_v1_prediction": row.get("parsed_prediction", ""),
            "parser_v1_mode": row.get("parser_mode", ""),
            "parser_v1_correct": v1_correct,
            "parser_v2_prediction": "" if v2_prediction is None else v2_prediction,
            "parser_v2_mode": v2_mode,
            "parser_v2_correct": v2_correct,
            "parser_v2_version": PARSER_VERSION,
            "parser_flip": flip,
            "parser_v2_numeric_token_count": len(NUMBER_RE.findall(raw.replace(",", ""))),
        }
    )
    return rescored


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["prompt_variant"])].append(row)

    summary = []
    for (model, prompt_variant), group in sorted(grouped.items()):
        total = len(group)
        v1_correct = sum(bool(row["parser_v1_correct"]) for row in group)
        v2_correct = sum(bool(row["parser_v2_correct"]) for row in group)
        v1_marker = sum(str(row["parser_v1_mode"]).endswith("marker") for row in group)
        v2_marker = sum(str(row["parser_v2_mode"]).endswith("marker") for row in group)
        v1_fallback = sum(row["parser_v1_mode"] == "last_number_fallback" for row in group)
        v2_fallback = sum(row["parser_v2_mode"] == "last_number_fallback" for row in group)
        multi_number = sum(int(row["parser_v2_numeric_token_count"]) >= 2 for row in group)
        incorrect_to_correct = sum(row["parser_flip"] == "incorrect_to_correct" for row in group)
        correct_to_incorrect = sum(row["parser_flip"] == "correct_to_incorrect" for row in group)
        summary.append(
            {
                "model": model,
                "prompt_variant": prompt_variant,
                "n": total,
                "parser_v1_accuracy": round(v1_correct / total, 6) if total else 0,
                "parser_v2_accuracy": round(v2_correct / total, 6) if total else 0,
                "accuracy_delta_v2_minus_v1": round((v2_correct - v1_correct) / total, 6)
                if total
                else 0,
                "parser_v1_marker_rate": round(v1_marker / total, 6) if total else 0,
                "parser_v2_marker_rate": round(v2_marker / total, 6) if total else 0,
                "parser_v1_fallback_rate": round(v1_fallback / total, 6) if total else 0,
                "parser_v2_fallback_rate": round(v2_fallback / total, 6) if total else 0,
                "multi_number_output_rate": round(multi_number / total, 6) if total else 0,
                "incorrect_to_correct_flips": incorrect_to_correct,
                "correct_to_incorrect_flips": correct_to_incorrect,
                "net_correct_flip": incorrect_to_correct - correct_to_incorrect,
                "parser_v2_version": PARSER_VERSION,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="results/prompt_rescue")
    parser.add_argument("--output-dir", default="results/prompt_rescue_rescore")
    args = parser.parse_args()

    input_root = ROOT / args.input_root
    output_dir = ROOT / args.output_dir
    output_files = discover_output_files(input_root)
    if not output_files:
        raise SystemExit(f"No prompt rescue outputs found under {input_root}")

    rescored_rows = []
    input_counts = {}
    for path in output_files:
        rows = read_jsonl(path)
        input_counts[str(path.relative_to(ROOT))] = len(rows)
        rescored_rows.extend(rescore_row(row) for row in rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / "parser_v2_outputs.jsonl"
    summary_path = output_dir / "parser_v2_summary.csv"
    metadata_path = output_dir / "parser_v2_run_metadata.json"
    write_jsonl(outputs_path, rescored_rows)
    write_csv(summary_path, summarize(rescored_rows))
    metadata = {
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parser_v2_version": PARSER_VERSION,
        "input_root": args.input_root,
        "input_files": input_counts,
        "n_rows": len(rescored_rows),
        "outputs": {
            "rescored_outputs": str(outputs_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "metadata": str(metadata_path.relative_to(ROOT)),
        },
        "note": (
            "Offline parser-only rescore of existing prompt rescue raw outputs. "
            "No model inference was run."
        ),
        "python": sys.version,
        "platform": platform.platform(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rescored rows={len(rescored_rows)} summary={summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
