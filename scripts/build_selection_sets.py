from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.selection.bias_audit import audit_selection_bias  # noqa: E402
from eg_sft.selection.error_guided import select_error_guided  # noqa: E402
from eg_sft.selection.matched_random import select_matched_random  # noqa: E402
from eg_sft.utils.io import read_csv, read_jsonl, write_csv, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    candidates_path = ROOT / "data" / "samples" / "candidate_pool.jsonl"
    profile_path = ROOT / "results" / "error_profile_v0.csv"
    if not candidates_path.exists():
        raise SystemExit("Missing candidate pool. Run: python scripts/generate_data.py --all")
    if not profile_path.exists():
        raise SystemExit("Missing error profile. Run: python scripts/run_base_diagnostic.py")

    candidates = read_jsonl(candidates_path)
    profile = read_csv(profile_path)
    targeted = select_error_guided(candidates, profile, budget=args.budget, seed=args.seed)
    matched = select_matched_random(candidates, targeted, seed=args.seed + 1)
    audit = audit_selection_bias(targeted, matched)

    out_dir = ROOT / "data" / "samples"
    write_jsonl(out_dir / f"selection_error_guided_b{args.budget}.jsonl", targeted)
    write_jsonl(out_dir / f"selection_matched_random_b{args.budget}.jsonl", matched)
    write_csv(ROOT / "results" / "selection_bias_audit.csv", audit)
    print(f"wrote targeted={len(targeted)} matched={len(matched)}")


if __name__ == "__main__":
    main()
