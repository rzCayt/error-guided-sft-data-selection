from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.selection.bias_audit import audit_selection_bias, summarize_selection_bias  # noqa: E402
from eg_sft.selection.error_guided import select_error_guided  # noqa: E402
from eg_sft.selection.matched_random import select_matched_random  # noqa: E402
from eg_sft.selection.strong_baselines import (  # noqa: E402
    select_exact_matched_random_multi_seed,
    select_metadata_hard_baseline,
    select_stratified_random,
    summarize_baseline_suite,
)
from eg_sft.utils.io import read_csv, read_jsonl, write_csv, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--baseline-seeds", default="20260711,20260712,20260713")
    parser.add_argument("--profile", default="results/error_profile_v0.csv")
    args = parser.parse_args()

    candidates_path = ROOT / "data" / "samples" / "candidate_pool.jsonl"
    profile_path = ROOT / args.profile
    if not candidates_path.exists():
        raise SystemExit("Missing candidate pool. Run: python scripts/generate_data.py --all")
    if not profile_path.exists():
        raise SystemExit("Missing error profile. Run: python scripts/run_base_diagnostic.py")

    candidates = read_jsonl(candidates_path)
    profile = read_csv(profile_path)
    targeted = select_error_guided(candidates, profile, budget=args.budget, seed=args.seed)
    matched = select_matched_random(candidates, targeted, seed=args.seed + 1)
    baseline_seeds = [int(seed.strip()) for seed in args.baseline_seeds.split(",") if seed.strip()]
    matched_many = select_exact_matched_random_multi_seed(candidates, targeted, baseline_seeds)
    stratified = select_stratified_random(candidates, budget=args.budget, seed=args.seed + 2)
    metadata_hard = select_metadata_hard_baseline(
        candidates,
        profile,
        budget=args.budget,
        seed=args.seed + 3,
    )
    strong_baselines = {
        **{f"exact_matched_random_seed_{seed}": rows for seed, rows in matched_many.items()},
        "stratified_random": stratified,
        "metadata_hard_baseline": metadata_hard,
    }
    audit = audit_selection_bias(targeted, matched)
    summary = summarize_selection_bias(targeted, matched)
    baseline_suite_summary = summarize_baseline_suite(targeted, strong_baselines)
    baseline_suite_summary.insert(
        0,
        {
            "baseline": "suite_metadata",
            "metric": "profile_path",
            "targeted": "",
            "baseline_value": args.profile,
            "delta": "",
            "note": (
                "Current default profile is a simulated diagnostic placeholder unless replaced "
                "with a real base diagnostic profile."
            ),
        },
    )

    out_dir = ROOT / "data" / "samples"
    write_jsonl(out_dir / f"selection_error_guided_b{args.budget}.jsonl", targeted)
    write_jsonl(out_dir / f"selection_matched_random_b{args.budget}.jsonl", matched)
    baseline_outputs = {}
    for name, rows in strong_baselines.items():
        relative_path = f"data/samples/baseline_{name}_b{args.budget}.jsonl"
        write_jsonl(ROOT / relative_path, rows)
        baseline_outputs[name] = relative_path
    write_csv(ROOT / "results" / "selection_bias_audit.csv", audit)
    write_csv(ROOT / "results" / "selection_bias_summary.csv", summary)
    write_csv(ROOT / "results" / "strong_baseline_suite_summary.csv", baseline_suite_summary)
    manifest = {
        "budget": args.budget,
        "profile_path": args.profile,
        "profile_is_default_simulated_placeholder": args.profile == "results/error_profile_v0.csv",
        "baseline_seeds": baseline_seeds,
        "baseline_outputs": baseline_outputs,
        "summary_path": "results/strong_baseline_suite_summary.csv",
        "created_by_command": " ".join(sys.argv),
        "note": (
            "Baselines built from results/error_profile_v0.csv are simulated-profile plumbing "
            "artifacts. Claim-bearing stages must use results/real_error_profile.csv."
        ),
    }
    manifest_path = ROOT / "results" / "strong_baseline_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"wrote targeted={len(targeted)} matched={len(matched)} "
        f"strong_baselines={len(strong_baselines)}"
    )


if __name__ == "__main__":
    main()
