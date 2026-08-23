"""Ordered, crash-durable batching primitives for cloud-v2 generation."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def contiguous_record_batches(
    *,
    records: Sequence[dict[str, Any]],
    start_index: int,
    batch_size: int,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Return ordered contiguous batches beginning at a resumable prefix."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= start_index <= len(records):
        raise ValueError("start_index is outside the frozen records")
    return [
        (index, list(records[index : index + batch_size]))
        for index in range(start_index, len(records), batch_size)
    ]


def append_jsonl_rows_fsynced(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Append rows in order and durably flush each complete JSONL record."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


def ordered_record_ids(
    batches: Sequence[tuple[int, Sequence[dict[str, Any]]]],
) -> list[str]:
    """Expose the output order for audits without depending on GPU code."""

    return [str(row["record_id"]) for _, batch in batches for row in batch]
