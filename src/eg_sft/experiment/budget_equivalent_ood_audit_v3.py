"""Fail-closed helpers for resumable, result-masked OOD audit publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
    ).encode("utf-8")


def write_bytes_exclusive_or_verify(path: Path, content: bytes) -> None:
    """Create an artifact once, or verify that a recovery attempt is identical."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing audit artifact differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def masked_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Keep only structural evidence; never persist result values before unblinding."""

    return {
        "status": "PASS",
        "record_count": int(report["record_count"]),
        "unique_record_id_count": int(report["unique_record_id_count"]),
        "parser_rows_recomputed_from_raw_text": int(
            report["parser_rows_recomputed_from_raw_text"]
        ),
        "ordered_frozen_membership": bool(report["ordered_frozen_membership"]),
        "gold_hashes_match": bool(report["gold_hashes_match"]),
        "accuracy_withheld": True,
    }
