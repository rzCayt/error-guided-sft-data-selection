"""Audit the official precomputed Qwen3/GSM8K plotting artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any


OFFICIAL_COMMIT = "c1dc0286b06b9e2a857d925e516938f4c9619dc2"
MODEL_DIR = Path("assets/plot_data/quantile_budget/qwen3-4b-base")
CSV_FILES = (
    "budget_true_metric_zero_shot.csv",
    "budget_true_metric_budget.csv",
    "budget_ce_loss_zero_shot.csv",
    "budget_ce_loss_budget.csv",
    "binning_true_metric_quantile.csv",
    "binning_ce_loss_quantile.csv",
)
PRIMARY_METHODS = ("Random", "RDS+ (RR)", "LESS (RR)")
EXPECTED_BUDGET_COLUMNS = {
    "dataset",
    "method",
    "num_samples",
    "seed",
    "true_metric",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head_commit(repo: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _read_primary_rows(path: Path, *, budget: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != EXPECTED_BUDGET_COLUMNS:
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            if (
                row["dataset"] == "gsm8k"
                and int(row["num_samples"]) == budget
                and row["method"] in PRIMARY_METHODS
            ):
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "num_samples": int(row["num_samples"]),
                        "seed": int(row["seed"]),
                        "true_metric": float(row["true_metric"]),
                    }
                )

    expected_pairs = {(method, seed) for method in PRIMARY_METHODS for seed in (0, 1, 2)}
    actual_pairs = {(row["method"], row["seed"]) for row in rows}
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"incomplete primary rows: expected {sorted(expected_pairs)}, "
            f"found {sorted(actual_pairs)}"
        )
    return sorted(rows, key=lambda row: (row["method"], row["seed"]))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(float(row["true_metric"]))
    return {
        method: {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
        for method, values in sorted(grouped.items())
    }


def reproduce_qwen3_gsm8k_artifact(
    *,
    official_repo: Path,
    output_dir: Path,
    budget: int = 500,
) -> dict[str, Any]:
    """Verify source commit/CSV hashes and extract the fixed GSM8K comparison."""

    commit = _head_commit(official_repo)
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"official repo commit {commit} != pinned {OFFICIAL_COMMIT}")

    model_dir = official_repo / MODEL_DIR
    hashes: dict[str, str] = {}
    for filename in CSV_FILES:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[filename] = _sha256(path)

    rows = _read_primary_rows(
        model_dir / "budget_true_metric_budget.csv", budget=budget
    )
    summary = {
        "artifact_type": "official_precomputed_plot_data_reproduction",
        "not_a_training_rerun": True,
        "official_commit": commit,
        "model": "Qwen/Qwen3-4B-Base",
        "dataset": "gsm8k",
        "budget": budget,
        "methods": list(PRIMARY_METHODS),
        "csv_sha256": hashes,
        "metric_summary": summarize_rows(rows),
        "claim_boundary": (
            "These values reproduce rows released by the official repository; "
            "they are not results from a new local training run."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "gsm8k_budget_rows.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
