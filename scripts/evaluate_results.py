from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.utils.io import read_csv, write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", default="results/base_diagnostic_results.csv")
    args = parser.parse_args()

    diag_path = ROOT / args.diagnostic
    if not diag_path.exists():
        raise SystemExit("Missing diagnostics. Run: python scripts/run_base_diagnostic.py")

    rows = read_csv(diag_path)
    total = len(rows)
    correct = sum(row["numeric_accuracy"] == "True" for row in rows)
    parse_success = sum(row["parse_success"] == "True" for row in rows)
    main_results = [
        {
            "condition": "base_simulated",
            "split": "dev_diagnostic",
            "n": total,
            "parse_success_rate": round(parse_success / total, 4) if total else 0,
            "numeric_accuracy": round(correct / total, 4) if total else 0,
            "note": "Simulated diagnostic placeholder; replace with real model outputs before claiming results.",
        },
        {
            "condition": "matched_random_lora",
            "split": "locked_tests",
            "n": 0,
            "parse_success_rate": "",
            "numeric_accuracy": "",
            "note": "Pending real LoRA run.",
        },
        {
            "condition": "error_guided_lora",
            "split": "locked_tests",
            "n": 0,
            "parse_success_rate": "",
            "numeric_accuracy": "",
            "note": "Pending real LoRA run.",
        },
    ]
    write_csv(ROOT / "results" / "main_results_v0.csv", main_results)
    print("wrote results/main_results_v0.csv")


if __name__ == "__main__":
    main()
