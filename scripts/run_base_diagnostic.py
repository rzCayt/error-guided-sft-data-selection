from __future__ import annotations

import argparse
import random
from collections import defaultdict

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.eval.error_taxonomy import classify_error  # noqa: E402
from eg_sft.eval.metrics import numeric_equal  # noqa: E402
from eg_sft.eval.parser import parse_numeric  # noqa: E402
from eg_sft.utils.io import read_jsonl, write_csv  # noqa: E402


def simulate_prediction(row: dict, rng: random.Random) -> str:
    family = row["task_family"]
    answer = float(row["answer"])
    difficulty = row["buckets"]["difficulty_bucket"]
    fail_prob = {"easy": 0.18, "medium": 0.32, "hard": 0.48}[difficulty]
    if family == "temporal_numeric_constraint":
        fail_prob += 0.12
    if family == "weighted_aggregation":
        fail_prob += 0.08

    if rng.random() > fail_prob:
        return f"The answer is {answer}."

    mode = rng.choice(["arithmetic", "formula", "scale", "parse", "binding"])
    if mode == "parse":
        return "I cannot determine the value from the given information."
    if mode == "scale":
        return f"The answer is {round(answer * rng.choice([10, 0.1, 100]), 4)}."
    if mode == "formula":
        return f"Using an average/sum shortcut, the answer is {round(answer * rng.uniform(0.55, 1.55), 4)}."
    if mode == "binding":
        return f"The answer is {round(answer + rng.choice([-17, -9, 11, 23]), 4)}."
    return f"The answer is {round(answer + rng.uniform(-4, 4), 4)}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/samples/dev_diagnostic.jsonl")
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    path = ROOT / args.input
    if not path.exists():
        raise SystemExit("Missing dev diagnostic data. Run: python scripts/generate_data.py --all")

    rng = random.Random(args.seed)
    rows = read_jsonl(path)
    diagnostics = []
    grouped = defaultdict(lambda: {"count": 0, "failures": 0})
    for row in rows:
        prediction = simulate_prediction(row, rng)
        parsed = parse_numeric(prediction)
        correct = numeric_equal(parsed, float(row["answer"]))
        error_type = classify_error(row["task_family"], float(row["answer"]), parsed, prediction)
        buckets = row["buckets"]
        diag = {
            "id": row["id"],
            "split": row["split"],
            "task_family": row["task_family"],
            "difficulty_bucket": buckets["difficulty_bucket"],
            "answer_magnitude_bucket": buckets["answer_magnitude_bucket"],
            "reasoning_length_bucket": buckets["reasoning_length_bucket"],
            "answer": row["answer"],
            "prediction": prediction,
            "parsed_prediction": "" if parsed is None else parsed,
            "parse_success": parsed is not None,
            "numeric_accuracy": correct,
            "output_length": len(prediction.split()),
            "error_type": error_type,
        }
        diagnostics.append(diag)
        key = (
            row["task_family"],
            buckets["difficulty_bucket"],
            buckets["answer_magnitude_bucket"],
            buckets["reasoning_length_bucket"],
        )
        grouped[key]["count"] += 1
        grouped[key]["failures"] += int(not correct)

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
                "error_rate": round(failures / count, 4) if count else 0,
            }
        )

    write_csv(ROOT / "results" / "base_diagnostic_results.csv", diagnostics)
    write_csv(ROOT / "results" / "error_profile_v0.csv", profile)
    accuracy = sum(row["numeric_accuracy"] for row in diagnostics) / len(diagnostics)
    print(f"wrote diagnostics={len(diagnostics)} accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
