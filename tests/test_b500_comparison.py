from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eg_sft.experiment.b500_comparison import (
    SEEDS,
    STRATEGIES,
    analyze_complete_matrix,
    load_complete_matrix,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_fake_run(
    root: Path,
    *,
    strategy: str,
    seed: int,
    correct_ids: set[int],
    example_count: int,
) -> None:
    run_id = f"fake_{strategy}_{seed}"
    run_dir = root / run_id
    raw_path = run_dir / "evaluation" / "raw_outputs.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "record_id": f"item-{index}",
            "numeric_correct": index in correct_ids,
            "strict_parse_status": "ok",
        }
        for index in range(example_count)
    ]
    raw_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(raw_text)
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    metrics = {
        "strategy": strategy,
        "seed": seed,
        "example_count": example_count,
        "numeric_correct_count": len(correct_ids),
        "numeric_accuracy": len(correct_ids) / example_count,
        "strict_parse_rate": 1.0,
        "raw_outputs_sha256": raw_sha,
    }
    metrics_path = run_dir / "evaluation" / "metrics.json"
    _write_json(metrics_path, metrics)
    _write_json(
        run_dir / "manifest.json",
        {"run_id": run_id, "seed": seed, "config": {"strategy": strategy}},
    )
    _write_json(
        run_dir / "run_complete.json",
        {
            "status": "PASS",
            "next_job_started": False,
            "raw_outputs_sha256": raw_sha,
            "evaluation_metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        },
    )
    _write_json(
        run_dir / "audits" / "formal_audit_v1.json",
        {
            "status": "PASS",
            "source_run_id": run_id,
            "strategy": strategy,
            "seed": seed,
        },
    )


def _fake_matrix(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "eg_sft.experiment.b500_comparison.EXPECTED_EXAMPLE_COUNT", 10
    )
    for seed in SEEDS:
        _write_fake_run(
            root,
            strategy="random",
            seed=seed,
            correct_ids={0, 1, 2, 3, 4},
            example_count=10,
        )
        _write_fake_run(
            root,
            strategy="rds_all",
            seed=seed,
            correct_ids={0, 1, 2, 3, 4, 5},
            example_count=10,
        )
        error_ids = {0, 1, 2, 3, 4, 5, 6} if seed != 41 else {0, 1, 2, 3, 4}
        _write_fake_run(
            root,
            strategy="rds_error",
            seed=seed,
            correct_ids=error_ids,
            example_count=10,
        )


def test_complete_matrix_and_frozen_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_matrix(tmp_path, monkeypatch)
    matrix = load_complete_matrix(tmp_path)
    assert set(matrix) == {(strategy, seed) for strategy in STRATEGIES for seed in SEEDS}

    report = analyze_complete_matrix(
        matrix,
        bootstrap_replicates=200,
        bootstrap_seed=123,
    )

    gate = report["frozen_downstream_gate"]
    assert gate["observed_positive_paired_seeds"] == 2
    assert gate["observed_mean_accuracy_difference"] == pytest.approx(1 / 30)
    assert gate["passed"] is True
    assert report["item_by_seed_direction_counts"] == {
        "rds_error_wins": 2,
        "rds_error_losses": 1,
        "ties": 27,
    }


def test_incomplete_matrix_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_matrix(tmp_path, monkeypatch)
    missing = tmp_path / "fake_rds_error_41" / "manifest.json"
    missing.rename(missing.with_suffix(".missing"))

    with pytest.raises(ValueError, match="missing="):
        load_complete_matrix(tmp_path)


def test_run_without_passing_audit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_matrix(tmp_path, monkeypatch)
    audit = tmp_path / "fake_random_17" / "audits" / "formal_audit_v1.json"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    _write_json(audit, payload)

    with pytest.raises(ValueError, match="no passing formal audit"):
        load_complete_matrix(tmp_path)
