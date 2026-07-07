from __future__ import annotations

from itertools import combinations

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.generator import SPLIT_SIZES, example_signature  # noqa: E402
from eg_sft.utils.io import read_jsonl, write_csv  # noqa: E402


def main() -> None:
    split_rows = {}
    for split in SPLIT_SIZES:
        path = ROOT / "data" / "samples" / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit("Missing generated data. Run: python scripts/generate_data.py --all")
        split_rows[split] = read_jsonl(path)

    rows = []
    for left, right in combinations(SPLIT_SIZES, 2):
        left_rows = split_rows[left]
        right_rows = split_rows[right]
        left_prompts = {row["prompt"] for row in left_rows}
        right_prompts = {row["prompt"] for row in right_rows}
        left_signatures = {example_signature(row) for row in left_rows}
        right_signatures = {example_signature(row) for row in right_rows}
        rows.append(
            {
                "left_split": left,
                "right_split": right,
                "left_count": len(left_rows),
                "right_count": len(right_rows),
                "exact_prompt_overlap": len(left_prompts & right_prompts),
                "exact_signature_overlap": len(left_signatures & right_signatures),
                "verdict": "pass" if not (left_signatures & right_signatures) else "fail",
            }
        )

    write_csv(ROOT / "results" / "split_leakage_audit.csv", rows)
    failures = [row for row in rows if row["verdict"] == "fail"]
    print(f"wrote results/split_leakage_audit.csv failures={len(failures)}")
    if failures:
        raise SystemExit("Split leakage audit failed")


if __name__ == "__main__":
    main()
