"""CPU-only state, schedule and recovery controls for the Phase-2 v7 run.

The event log is append-only. ``current_state.json`` is only a materialized
view of that log, so an interrupted atomic replacement never erases the
transition history.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from eg_sft.evaluation.phase2_v7_canary import file_sha256
from eg_sft.experiment.phase2_crossed_v7 import validate_phase2_matrix


STATES = ("PLANNED", "LOCKED", "RUNNING", "AUDITING", "COMPLETE", "FAILED")
ALLOWED_TRANSITIONS = {
    "PLANNED": {"LOCKED"},
    "LOCKED": {"RUNNING", "FAILED"},
    "RUNNING": {"AUDITING", "FAILED"},
    "AUDITING": {"COMPLETE", "FAILED"},
    "FAILED": {"LOCKED"},
    "COMPLETE": set(),
}


def _validate_long_matrix(payload: dict[str, Any]) -> None:
    version = payload.get("matrix_version")
    if version == "phase2-crossed-48cell-v7":
        validate_phase2_matrix(payload)
        return
    if version == "phase2-clean-common24-v8":
        from eg_sft.experiment.phase2_clean_common_v8 import (
            validate_clean_common_matrix,
        )

        validate_clean_common_matrix(payload)
        return
    raise ValueError(f"unsupported long-experiment matrix: {version}")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _advisory_lock(path: Path) -> Iterator[BinaryIO]:
    """Cross-platform advisory lock whose file persists after release."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def worker_schedule(matrix: Mapping[str, Any], worker_id: str) -> list[str]:
    _validate_long_matrix(dict(matrix))
    if worker_id not in {"gpu0", "gpu1"}:
        raise ValueError("worker_id must be gpu0 or gpu1")
    cells = [str(wave[worker_id]) for wave in matrix["dual_gpu_schedule"]]
    known = {str(row["cell_id"]) for row in matrix["job_order"]}
    other = "gpu1" if worker_id == "gpu0" else "gpu0"
    other_cells = [str(wave[other]) for wave in matrix["dual_gpu_schedule"]]
    if (
        not cells
        or len(cells) != len(set(cells))
        or any(cell not in known for cell in cells)
        or set(cells) & set(other_cells)
    ):
        raise ValueError("worker schedule is incomplete, duplicated or overlapping")
    return cells


class Phase2StateStore:
    def __init__(self, *, root: Path, matrix_path: Path) -> None:
        self.root = root.resolve()
        self.matrix_path = matrix_path.resolve()
        self.matrix = read_json(self.matrix_path)
        _validate_long_matrix(self.matrix)
        self.matrix_sha256 = file_sha256(self.matrix_path)
        self.by_cell = {
            str(row["cell_id"]): dict(row) for row in self.matrix["job_order"]
        }
        self.worker_by_cell = {
            cell_id: worker
            for worker in ("gpu0", "gpu1")
            for cell_id in worker_schedule(self.matrix, worker)
        }

    def cell_dir(self, cell_id: str) -> Path:
        if cell_id not in self.by_cell:
            raise ValueError(f"unknown Phase-2 cell: {cell_id}")
        return self.root / "cells" / cell_id

    def state_path(self, cell_id: str) -> Path:
        return self.cell_dir(cell_id) / "current_state.json"

    def read_state(self, cell_id: str) -> dict[str, Any]:
        path = self.state_path(cell_id)
        if not path.is_file():
            raise ValueError(f"cell has not been initialized: {cell_id}")
        state = read_json(path)
        if (
            state.get("cell_id") != cell_id
            or state.get("matrix_sha256") != self.matrix_sha256
            or state.get("state") not in STATES
        ):
            raise ValueError(f"cell state contract changed: {cell_id}")
        return state

    def initialize(self) -> dict[str, Any]:
        created = 0
        for cell_id, job in self.by_cell.items():
            cell_dir = self.cell_dir(cell_id)
            with _advisory_lock(cell_dir / "state.lock"):
                path = self.state_path(cell_id)
                if path.is_file():
                    self.read_state(cell_id)
                    continue
                event_id = uuid.uuid4().hex
                state = {
                    "schema_version": "phase2-v7-cell-state-v1",
                    "cell_id": cell_id,
                    "method": job["method"],
                    "replicate_index": int(job["replicate_index"]),
                    "train_seed": int(job["train_seed"]),
                    "worker_id": self.worker_by_cell[cell_id],
                    "matrix_sha256": self.matrix_sha256,
                    "state": "PLANNED",
                    "attempt_id": None,
                    "transition_index": 0,
                    "event_id": event_id,
                    "updated_at_utc": utc_now(),
                    "accuracy_withheld": True,
                }
                _append_fsynced(
                    cell_dir / "events.jsonl",
                    canonical_json_bytes(state | {"event": "state_initialized"}),
                )
                _atomic_write(path, canonical_json_bytes(state))
                created += 1
        return {
            "status": "PASS",
            "job_count": len(self.by_cell),
            "created_count": created,
            "existing_count": len(self.by_cell) - created,
            "matrix_sha256": self.matrix_sha256,
            "gpu_accessed": False,
            "accuracy_withheld": True,
        }

    def transition(
        self,
        *,
        cell_id: str,
        target: str,
        worker_id: str,
        reason: str,
        attempt_id: str | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if target not in STATES or target == "PLANNED":
            raise ValueError(f"invalid transition target: {target}")
        if worker_id != self.worker_by_cell.get(cell_id):
            raise ValueError("worker is not assigned to this cell")
        cell_dir = self.cell_dir(cell_id)
        with _advisory_lock(cell_dir / "state.lock"):
            current = self.read_state(cell_id)
            source = str(current["state"])
            if target not in ALLOWED_TRANSITIONS[source]:
                raise ValueError(f"forbidden state transition: {source}->{target}")
            current_attempt = current.get("attempt_id")
            if target == "LOCKED":
                if attempt_id is None:
                    attempt_id = uuid.uuid4().hex
                if attempt_id == current_attempt:
                    raise ValueError("recovery must use a new attempt_id")
            elif not current_attempt or attempt_id != current_attempt:
                raise ValueError("state transition attempt_id changed")
            event_id = uuid.uuid4().hex
            updated = {
                **current,
                "state": target,
                "attempt_id": attempt_id,
                "transition_index": int(current["transition_index"]) + 1,
                "event_id": event_id,
                "updated_at_utc": utc_now(),
                "reason": reason,
            }
            if evidence:
                updated["evidence"] = dict(evidence)
            event = {
                "schema_version": "phase2-v7-cell-event-v1",
                "event_id": event_id,
                "cell_id": cell_id,
                "worker_id": worker_id,
                "matrix_sha256": self.matrix_sha256,
                "source_state": source,
                "target_state": target,
                "attempt_id": attempt_id,
                "transition_index": updated["transition_index"],
                "reason": reason,
                "evidence": dict(evidence or {}),
                "recorded_at_utc": updated["updated_at_utc"],
                "accuracy_withheld": True,
            }
            _append_fsynced(cell_dir / "events.jsonl", canonical_json_bytes(event))
            _atomic_write(self.state_path(cell_id), canonical_json_bytes(updated))
            return updated

    def fail(
        self,
        *,
        cell_id: str,
        worker_id: str,
        attempt_id: str,
        reason: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.transition(
            cell_id=cell_id,
            target="FAILED",
            worker_id=worker_id,
            reason=reason,
            attempt_id=attempt_id,
            evidence=evidence,
        )

    def registry(self) -> dict[str, Any]:
        rows = []
        counts = {state: 0 for state in STATES}
        for cell_id in self.by_cell:
            state = self.read_state(cell_id)
            counts[str(state["state"])] += 1
            rows.append(state)
        return {
            "schema_version": "phase2-v7-control-registry-v1",
            "status": "PASS",
            "matrix_sha256": self.matrix_sha256,
            "job_count": len(rows),
            "state_counts": counts,
            "cells": rows,
            "accuracy_withheld": True,
            "gpu_accessed": False,
        }


def validate_complete_evidence(
    *, state: Mapping[str, Any], required_hashes: Sequence[str]
) -> None:
    if state.get("state") != "COMPLETE":
        raise ValueError("cell is not COMPLETE")
    evidence = state.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("COMPLETE state lacks evidence")
    for field in required_hashes:
        value = str(evidence.get(field, ""))
        if len(value) != 64:
            raise ValueError(f"COMPLETE evidence hash is missing: {field}")
