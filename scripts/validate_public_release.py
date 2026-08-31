"""Mechanically validate the professor-facing release and its bounded claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "results" / "public_release_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def validate(bundle: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hash_failures = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != entry["sha256"]:
            hash_failures.append(
                {"path": entry["path"], "expected": entry["sha256"], "actual": actual}
            )
    checks.append(
        {
            "id": "manifest_hashes",
            "passed": not hash_failures,
            "details": hash_failures or f"{len(manifest['files'])} file hashes match",
        }
    )

    selector = json.loads(
        (bundle / "selector_identifiability_rerun" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    selector_passed = (
        selector["identifiability_status"]
        == "not_identifiable_beyond_matching_metadata"
        and selector["candidate_count"] == 500
        and selector["budget"] == 128
        and selector["training_allowed"] is False
    )
    checks.append(
        {
            "id": "metadata_selector_claim",
            "passed": selector_passed,
            "details": {
                "status": selector["identifiability_status"],
                "candidate_count": selector["candidate_count"],
                "budget": selector["budget"],
                "training_allowed": selector["training_allowed"],
            },
        }
    )

    residual = json.loads(
        (ROOT / "results" / "residual_selector_identifiability" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    residual_passed = (
        residual["identifiability_status"]
        == "not_identifiable_beyond_static_operation_metadata"
        and residual["candidate_level_variation_beyond_operation_signature"] is False
        and residual["test_split_accessed"] is False
        and residual["gold_test_information_used"] is False
        and residual["candidate_answer_or_rationale_used"] is False
    )
    checks.append(
        {
            "id": "residual_selector_claim",
            "passed": residual_passed,
            "details": {
                "status": residual["identifiability_status"],
                "candidate_level_variation": residual[
                    "candidate_level_variation_beyond_operation_signature"
                ],
                "test_split_accessed": residual["test_split_accessed"],
            },
        }
    )

    f2 = json.loads(
        (ROOT / "results" / "model_aware_signal_f2" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    observed_delta = float(f2["observed"]["t_delta"])
    p90_delta = float(f2["permutation_p90"]["t_delta"])
    f2_passed = (
        f2["status"] == "frozen_negative_result"
        and f2["sample_counts"]["candidate"] == 8
        and observed_delta < p90_delta
        and "t_delta_ge_permutation_p90" in f2["failed_gates"]
    )
    checks.append(
        {
            "id": "model_aware_f2_claim",
            "passed": f2_passed,
            "details": {
                "candidate_count": f2["sample_counts"]["candidate"],
                "observed_delta": observed_delta,
                "permutation_p90_delta": p90_delta,
                "failed_gates": f2["failed_gates"],
            },
        }
    )

    pipeline_dir = bundle / "model_pipeline_check_25"
    raw_rows = [
        json.loads(line)
        for line in (pipeline_dir / "scale_model_diagnostic_outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    summary = read_single_csv(pipeline_dir / "scale_model_diagnostic_summary.csv")
    numeric_correct = sum(bool(row["numeric_accuracy"]) for row in raw_rows)
    parsed = sum(bool(row["parse_success"]) for row in raw_rows)
    pipeline_passed = (
        len(raw_rows) == 25
        and numeric_correct == 19
        and parsed == 25
        and float(summary["numeric_accuracy"]) == numeric_correct / len(raw_rows)
        and float(summary["parse_success_rate"]) == parsed / len(raw_rows)
        and {row["model"] for row in raw_rows} == {"Qwen/Qwen3-1.7B"}
    )
    checks.append(
        {
            "id": "model_pipeline_claim",
            "passed": pipeline_passed,
            "details": {
                "n": len(raw_rows),
                "numeric_correct": numeric_correct,
                "parsed": parsed,
                "split": sorted({row["split"] for row in raw_rows}),
            },
        }
    )

    overall = all(bool(check["passed"]) for check in checks)
    return {
        "schema_version": 1,
        "overall_passed": overall,
        "claim_scope": (
            "Artifact existence, hash integrity, selector negative results, and the bounded "
            "25-item pipeline metrics only. No SFT-effectiveness claim is tested."
        ),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    report = validate(args.bundle.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
