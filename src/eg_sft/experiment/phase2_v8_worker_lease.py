"""Process-level worker lease with a persistent 60-second heartbeat."""

from __future__ import annotations

import json
import os
import socket
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO


class WorkerLease:
    def __init__(
        self, *, root: Path, worker_id: str, gpu_uuid: str, heartbeat_seconds: int = 60
    ) -> None:
        self.root = root.resolve()
        self.worker_id = worker_id
        self.gpu_uuid = gpu_uuid
        self.heartbeat_seconds = heartbeat_seconds
        self._handle: BinaryIO | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _write_heartbeat(self) -> None:
        payload = {
            "schema_version": "phase2-v8-worker-lease-v1",
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "gpu_uuid": self.gpu_uuid,
            "heartbeat_at_utc": datetime.now(UTC).isoformat(),
        }
        path = self.root / "worker_lease.json"
        temporary = self.root / f".worker_lease.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self._write_heartbeat()

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        handle = (self.root / "worker_process.lock").open("a+b")
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                handle.close()
                raise RuntimeError("v8 worker lease is held by a live process") from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                handle.close()
                raise RuntimeError("v8 worker lease is held by a live process") from error
        self._handle = handle
        self._write_heartbeat()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._handle is not None:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
