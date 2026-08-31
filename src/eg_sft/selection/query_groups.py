"""Freeze auditable all-query and error-query groups from diagnostic outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _canonical_sha256(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(
        list(rows),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_unique(
    rows: Sequence[dict[str, Any]],
    *,
    id_field: str,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get(id_field, ""))
        if not record_id:
            raise ValueError(f"{source_name} contains a row without {id_field}")
        if record_id in indexed:
            raise ValueError(f"{source_name} contains duplicate ID: {record_id}")
        indexed[record_id] = row
    return indexed


def freeze_query_groups(
    *,
    split_records: Sequence[dict[str, Any]],
    diagnostic_outputs: Sequence[dict[str, Any]],
    protocol_split: str = "selection_diagnostic",
    minimum_group_size: int = 64,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate diagnostic coverage, then freeze text-free query lists.

    The returned rows contain only stable IDs, source indices, question hashes,
    and the observed correctness label. Query text is reloaded later from the
    pinned dataset revision rather than copied into the public artifact.
    """

    if minimum_group_size <= 0:
        raise ValueError("minimum_group_size must be positive")

    expected_rows = [
        row for row in split_records if row.get("protocol_split") == protocol_split
    ]
    expected = _index_unique(
        expected_rows,
        id_field="record_id",
        source_name="split records",
    )
    observed = _index_unique(
        diagnostic_outputs,
        id_field="record_id",
        source_name="diagnostic outputs",
    )

    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "diagnostic coverage mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    all_queries: list[dict[str, Any]] = []
    for record_id in sorted(expected):
        split_row = expected[record_id]
        output_row = observed[record_id]
        if output_row.get("question_sha256") != split_row.get("question_sha256"):
            raise ValueError(f"question hash mismatch for {record_id}")
        if output_row.get("parse_status") != "ok":
            raise ValueError(f"unresolved diagnostic output for {record_id}")
        numeric_correct = output_row.get("numeric_correct")
        if not isinstance(numeric_correct, bool):
            raise ValueError(f"numeric_correct must be boolean for {record_id}")

        all_queries.append(
            {
                "record_id": record_id,
                "source_index": int(split_row["source_index"]),
                "question_sha256": str(split_row["question_sha256"]),
                "numeric_correct": numeric_correct,
            }
        )

    error_queries = [row for row in all_queries if not row["numeric_correct"]]
    correct_count = len(all_queries) - len(error_queries)
    gate_passed = (
        len(error_queries) >= minimum_group_size
        and correct_count >= minimum_group_size
    )
    manifest = {
        "protocol_split": protocol_split,
        "all_query_count": len(all_queries),
        "correct_query_count": correct_count,
        "error_query_count": len(error_queries),
        "minimum_group_size": minimum_group_size,
        "group_size_gate_passed": gate_passed,
        "fallback_required": not gate_passed,
        "all_queries_sha256": _canonical_sha256(all_queries),
        "error_queries_sha256": _canonical_sha256(error_queries),
    }
    return all_queries, error_queries, manifest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
