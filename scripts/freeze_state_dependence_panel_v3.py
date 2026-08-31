#!/usr/bin/env python3
"""Freeze a universal-unseen 48-candidate panel for state dependence v3.

The builder uses only frozen candidate scores, source labels, target adapter
selection manifests, and deterministic hashes.  It never reads utility
outcomes.  It fails if any target manifest is missing or hash-mismatched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _quartile(rank: int) -> int:
    if rank < 0 or rank >= 96:
        raise ValueError(f"error_query_rank is outside [0, 95]: {rank}")
    return rank // 24


def _manifest_selected_ids(manifest: dict[str, Any], path: Path) -> list[str]:
    rows = manifest.get("selected_candidates")
    if not isinstance(rows, list) or len(rows) != 500:
        raise ValueError(f"{path}: expected 500 selected_candidates")
    values = [str(row["candidate_id"]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: duplicate selected candidate IDs")
    return values


def freeze_panel(
    *,
    protocol_path: Path,
    candidate_scores_path: Path,
    adapter_index_path: Path,
    manifest_root: Path,
    output_panel_path: Path,
    output_overlap_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for output in (output_panel_path, output_overlap_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
    protocol = read_json(protocol_path)
    adapter_index = read_json(adapter_index_path)
    target_states = list(
        protocol["stage_u1_cross_state_transfer"]["initial_adapter_states"]
    )
    index_by_state = {
        str(row["cell_id"]): row for row in adapter_index.get("adapters", [])
    }
    if not set(target_states) <= set(index_by_state):
        raise ValueError("adapter index does not contain every initial state")

    manifest_rows: list[dict[str, Any]] = []
    selected_by_state: dict[str, set[str]] = {}
    for state_id in target_states:
        index_row = index_by_state[state_id]
        method = str(index_row["method"])
        replicate = int(index_row["replicate_index"])
        path = (
            manifest_root
            / f"replicate_{replicate:02d}"
            / method
            / "selection_manifest.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing recovered selection manifest: {path}")
        observed_sha = file_sha256(path)
        expected_sha = str(index_row["selection_manifest_sha256"])
        if observed_sha != expected_sha:
            raise ValueError(f"{state_id}: selection manifest SHA mismatch")
        selected_ids = _manifest_selected_ids(read_json(path), path)
        selected_by_state[state_id] = set(selected_ids)
        manifest_rows.append(
            {
                "state_id": state_id,
                "method": method,
                "replicate_index": replicate,
                "selection_manifest_sha256": observed_sha,
                "selected_id_sha256": id_sha256(selected_ids),
                "selected_count": len(selected_ids),
            }
        )

    scores = read_jsonl(candidate_scores_path)
    if len(scores) != 96:
        raise ValueError(f"expected 96 candidate score rows, found {len(scores)}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in scores:
        candidate_id = str(row["candidate_id"])
        if candidate_id in by_id:
            raise ValueError(f"duplicate candidate score row: {candidate_id}")
        if row.get("response_only_trainable") is not True:
            raise ValueError(f"untrainable candidate in score panel: {candidate_id}")
        by_id[candidate_id] = row

    union_seen = set().union(*selected_by_state.values())
    unseen_rows = [row for row in scores if str(row["candidate_id"]) not in union_seen]
    seen_rows = [row for row in scores if str(row["candidate_id"]) in union_seen]
    unseen_by_quartile: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in unseen_rows:
        unseen_by_quartile[_quartile(int(row["error_query_rank"]))].append(row)
    if any(len(unseen_by_quartile[q]) < 12 for q in range(4)):
        counts = {q: len(unseen_by_quartile[q]) for q in range(4)}
        raise ValueError(f"not enough universal-unseen candidates per quartile: {counts}")

    chosen: list[dict[str, Any]] = []
    for quartile in range(4):
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in unseen_by_quartile[quartile]:
            by_source[str(row["source_dataset"])].append(row)
        for source, rows in by_source.items():
            rows.sort(
                key=lambda row: stable_hash(
                    "20260831_state_dependence_panel_v3",
                    str(quartile),
                    source,
                    str(row["candidate_id"]),
                )
            )
        quartile_rows: list[dict[str, Any]] = []
        round_index = 0
        while len(quartile_rows) < 12:
            sources = [source for source, rows in by_source.items() if rows]
            if not sources:
                raise ValueError(f"quartile {quartile} cannot fill 12 candidates")
            sources.sort(
                key=lambda source: stable_hash(
                    "20260831_state_dependence_source_round_v3",
                    str(quartile),
                    str(round_index),
                    source,
                )
            )
            for source in sources:
                if len(quartile_rows) == 12:
                    break
                quartile_rows.append(by_source[source].pop(0))
            round_index += 1
        chosen.extend(sorted(quartile_rows, key=lambda row: int(row["error_query_rank"])))

    panel_rows: list[dict[str, Any]] = []
    for row in chosen:
        candidate_id = str(row["candidate_id"])
        state_exposure = {
            state_id: candidate_id in selected_ids
            for state_id, selected_ids in selected_by_state.items()
        }
        if any(state_exposure.values()):
            raise AssertionError(f"selected candidate is not universal-unseen: {candidate_id}")
        panel_rows.append(
            {
                "candidate_id": candidate_id,
                "source_dataset": str(row["source_dataset"]),
                "source_index": int(row["source_index"]),
                "prompt_sha256": str(row["prompt_sha256"]),
                "error_query_rank": int(row["error_query_rank"]),
                "error_query_score": float(row["error_query_score"]),
                "all_query_rank": int(row["all_query_rank"]),
                "all_query_score": float(row["all_query_score"]),
                "training_supervised_tokens": int(row["training_supervised_tokens"]),
                "training_total_tokens": int(row["training_total_tokens"]),
                "error_score_quartile": _quartile(int(row["error_query_rank"])),
                "unseen_by_all_initial_states": True,
            }
        )
    candidate_ids = [str(row["candidate_id"]) for row in panel_rows]
    if len(candidate_ids) != 48 or len(set(candidate_ids)) != 48:
        raise AssertionError("v3 panel must contain 48 unique candidates")

    overlap = {
        "schema_version": "state-dependence-overlap-audit-v3",
        "status": "PASS",
        "candidate_scores_sha256": file_sha256(candidate_scores_path),
        "adapter_index_sha256": file_sha256(adapter_index_path),
        "manifest_root_recorded_as_external_input": True,
        "target_state_count": len(target_states),
        "target_manifests": manifest_rows,
        "target_training_union_count": len(union_seen),
        "score_panel_count": len(scores),
        "score_panel_seen_count": len(seen_rows),
        "score_panel_unseen_count": len(unseen_rows),
        "score_panel_seen_ids": sorted(str(row["candidate_id"]) for row in seen_rows),
        "unseen_counts_by_quartile": {
            f"q{q}": len(unseen_by_quartile[q]) for q in range(4)
        },
        "frozen_panel_overlap_count": 0,
        "frozen_panel_selected_id_sha256": id_sha256(candidate_ids),
    }
    panel = {
        "schema_version": "state-dependence-candidate-panel-v3",
        "status": "FROZEN_CPU_ONLY",
        "research_mode": "prospective_confirmatory_followup",
        "selection_rule": "universal_unseen_then_error_score_quartile_12_each_with_deterministic_source_round_robin",
        "utility_outcomes_read_by_builder": False,
        "historical_reliability_candidates_forced": False,
        "candidate_count": len(panel_rows),
        "quartile_counts": {f"q{q}": 12 for q in range(4)},
        "selected_id_sha256": id_sha256(candidate_ids),
        "candidate_scores_sha256": file_sha256(candidate_scores_path),
        "overlap_audit_sha256": "PENDING_WRITE_BINDING",
        "candidates": panel_rows,
    }
    output_overlap_path.parent.mkdir(parents=True, exist_ok=True)
    output_overlap_path.write_text(
        json.dumps(overlap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    panel["overlap_audit_sha256"] = file_sha256(output_overlap_path)
    output_panel_path.parent.mkdir(parents=True, exist_ok=True)
    output_panel_path.write_text(
        json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return panel, overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--adapter-index", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-panel", type=Path, required=True)
    parser.add_argument("--output-overlap", type=Path, required=True)
    args = parser.parse_args()
    panel, overlap = freeze_panel(
        protocol_path=args.protocol.resolve(),
        candidate_scores_path=args.candidate_scores.resolve(),
        adapter_index_path=args.adapter_index.resolve(),
        manifest_root=args.manifest_root.resolve(),
        output_panel_path=args.output_panel.resolve(),
        output_overlap_path=args.output_overlap.resolve(),
    )
    print(
        json.dumps(
            {
                "status": overlap["status"],
                "candidate_count": panel["candidate_count"],
                "selected_id_sha256": panel["selected_id_sha256"],
                "score_panel_seen_count": overlap["score_panel_seen_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
