from __future__ import annotations

from pathlib import Path

import pytest

from eg_sft.experiment.phase2_v8_worker_lease import WorkerLease


def test_live_worker_process_conflict_is_rejected(tmp_path: Path) -> None:
    first = WorkerLease(root=tmp_path, worker_id="gpu0", gpu_uuid="GPU-a")
    second = WorkerLease(root=tmp_path, worker_id="gpu0", gpu_uuid="GPU-a")
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="live process"):
            second.acquire()
        assert (tmp_path / "worker_lease.json").is_file()
    finally:
        first.close()
