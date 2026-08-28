"""Run one v4 cell with a worker-scoped lock instead of a matrix-global lock."""

from __future__ import annotations

import os
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import run_budget_equivalent_cell as implementation  # noqa: E402
import run_budget_equivalent_cell_v3 as public_v3  # noqa: E402
from run_b500_formal_resumable import _global_job_lock as original_lock  # noqa: E402


def resolve_worker_id(raw_value: str | None) -> str:
    """Use ``manual`` only when the worker environment variable is absent."""

    worker_id = "manual" if raw_value is None else raw_value.strip()
    if not worker_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in worker_id
    ):
        raise ValueError("EG_SFT_WORKER_ID must be a simple non-empty worker name")
    return worker_id


def main() -> None:
    worker_id = resolve_worker_id(os.environ.get("EG_SFT_WORKER_ID"))

    def worker_lock(output_root: Path):
        return original_lock(output_root / "worker_locks" / worker_id)

    implementation._global_job_lock = worker_lock
    public_v3.main()


if __name__ == "__main__":
    main()
