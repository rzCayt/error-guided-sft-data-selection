from __future__ import annotations

import json
from pathlib import Path

import pytest

from eg_sft.experiment.phase2_v7_control import (
    Phase2StateStore,
    validate_complete_evidence,
    worker_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs" / "phase2_crossed_48cell_v7.json"


def test_worker_schedule_is_dynamic_disjoint_and_complete() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    gpu0 = worker_schedule(matrix, "gpu0")
    gpu1 = worker_schedule(matrix, "gpu1")
    assert len(gpu0) == len(gpu1) == 16
    assert not set(gpu0) & set(gpu1)
    assert set(gpu0) | set(gpu1) == {
        row["cell_id"] for row in matrix["job_order"]
    }


def test_state_machine_records_recovery_without_overwriting_history(
    tmp_path: Path,
) -> None:
    store = Phase2StateStore(root=tmp_path / "control", matrix_path=MATRIX)
    report = store.initialize()
    assert report["job_count"] == 32
    cell = worker_schedule(store.matrix, "gpu0")[0]
    locked = store.transition(
        cell_id=cell,
        target="LOCKED",
        worker_id="gpu0",
        reason="test",
        attempt_id="attempt-a",
    )
    store.transition(
        cell_id=cell,
        target="RUNNING",
        worker_id="gpu0",
        reason="test",
        attempt_id="attempt-a",
    )
    store.fail(
        cell_id=cell,
        worker_id="gpu0",
        reason="interrupted",
        attempt_id="attempt-a",
    )
    recovered = store.transition(
        cell_id=cell,
        target="LOCKED",
        worker_id="gpu0",
        reason="explicit recovery",
        attempt_id="attempt-b",
    )
    assert locked["state"] == "LOCKED"
    assert recovered["attempt_id"] == "attempt-b"
    events = (store.cell_dir(cell) / "events.jsonl").read_text(encoding="utf-8")
    assert events.count("\n") == 5


def test_complete_is_terminal_and_requires_hash_evidence(tmp_path: Path) -> None:
    store = Phase2StateStore(root=tmp_path / "control", matrix_path=MATRIX)
    store.initialize()
    cell = worker_schedule(store.matrix, "gpu1")[0]
    attempt = "attempt-z"
    for target in ("LOCKED", "RUNNING", "AUDITING"):
        store.transition(
            cell_id=cell,
            target=target,
            worker_id="gpu1",
            reason="test",
            attempt_id=attempt,
        )
    complete = store.transition(
        cell_id=cell,
        target="COMPLETE",
        worker_id="gpu1",
        reason="audits passed",
        attempt_id=attempt,
        evidence={"formal_audit_sha256": "a" * 64, "ood_audit_sha256": "b" * 64},
    )
    validate_complete_evidence(
        state=complete,
        required_hashes=("formal_audit_sha256", "ood_audit_sha256"),
    )
    with pytest.raises(ValueError, match="forbidden"):
        store.transition(
            cell_id=cell,
            target="FAILED",
            worker_id="gpu1",
            reason="cannot rewrite complete",
            attempt_id=attempt,
        )


def test_wrong_worker_and_reused_attempt_fail_closed(tmp_path: Path) -> None:
    store = Phase2StateStore(root=tmp_path / "control", matrix_path=MATRIX)
    store.initialize()
    cell = worker_schedule(store.matrix, "gpu0")[0]
    with pytest.raises(ValueError, match="not assigned"):
        store.transition(
            cell_id=cell,
            target="LOCKED",
            worker_id="gpu1",
            reason="wrong worker",
            attempt_id="a",
        )
    store.transition(
        cell_id=cell,
        target="LOCKED",
        worker_id="gpu0",
        reason="first",
        attempt_id="a",
    )
    with pytest.raises(ValueError, match="attempt_id"):
        store.transition(
            cell_id=cell,
            target="RUNNING",
            worker_id="gpu0",
            reason="changed",
            attempt_id="b",
        )
