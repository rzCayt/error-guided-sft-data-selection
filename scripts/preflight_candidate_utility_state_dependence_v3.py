#!/usr/bin/env python3
"""CPU-only preflight for the frozen state-dependence v3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def preflight(
    *,
    protocol_path: Path,
    panel_path: Path,
    overlap_path: Path,
    adapter_index_path: Path,
    gsm8k_records_path: Path,
    tulu_pool_path: Path,
    formal_token_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    protocol = read_json(protocol_path)
    panel = read_json(panel_path)
    overlap = read_json(overlap_path)
    adapter_index = read_json(adapter_index_path)
    token_audit = read_json(formal_token_audit_path)

    if protocol.get("schema_version") != "candidate-utility-state-dependence-protocol-v3":
        raise ValueError("not a state-dependence v3 protocol")
    if panel.get("schema_version") != "state-dependence-candidate-panel-v3":
        raise ValueError("not a state-dependence v3 panel")
    if panel.get("utility_outcomes_read_by_builder") is not False:
        raise ValueError("candidate panel was not built independently of utility outcomes")
    if panel.get("historical_reliability_candidates_forced") is not False:
        raise ValueError("historical reliability candidates were forced into the v3 panel")
    if overlap.get("status") != "PASS" or int(overlap["frozen_panel_overlap_count"]) != 0:
        raise ValueError("candidate exposure audit did not pass")
    if file_sha256(overlap_path) != panel.get("overlap_audit_sha256"):
        raise ValueError("panel is not bound to the supplied overlap audit")
    if adapter_index.get("status") != "PASS":
        raise ValueError("adapter evidence index did not pass")

    candidate_rows = list(panel.get("candidates", []))
    candidate_ids = [str(row["candidate_id"]) for row in candidate_rows]
    if len(candidate_ids) != 48 or len(set(candidate_ids)) != 48:
        raise ValueError("v3 panel must contain 48 unique candidates")
    if id_sha256(candidate_ids) != panel.get("selected_id_sha256"):
        raise ValueError("v3 panel selected-ID hash changed")
    if any(row.get("unseen_by_all_initial_states") is not True for row in candidate_rows):
        raise ValueError("v3 panel contains a candidate not marked universal-unseen")

    tulu_rows = read_jsonl(tulu_pool_path)
    tulu_by_id = {str(row["candidate_id"]): row for row in tulu_rows}
    if len(tulu_by_id) != len(tulu_rows):
        raise ValueError("Tulu manifest contains duplicate candidate IDs")
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"])
        source = tulu_by_id.get(candidate_id)
        if source is None or source.get("prompt_sha256") != row["prompt_sha256"]:
            raise ValueError(f"candidate missing or changed in Tulu manifest: {candidate_id}")

    gsm_rows = read_jsonl(gsm8k_records_path)
    utility_records = sorted(
        (row for row in gsm_rows if row["protocol_split"] == "candidate_utility_validation"),
        key=lambda row: (int(row["source_index"]), str(row["record_id"])),
    )
    utility_ids = [str(row["record_id"]) for row in utility_records]
    token_utility_ids = [str(row["record_id"]) for row in token_audit["utility_examples"]]
    if len(utility_ids) != 128 or utility_ids != token_utility_ids:
        raise ValueError("utility-set IDs differ from the frozen token audit")

    initial_states = list(
        protocol["stage_u1_cross_state_transfer"]["initial_adapter_states"]
    )
    expansion_states = list(
        protocol["stage_u1_cross_state_transfer"]["expansion_adapter_states"]
    )
    indexed_states = {str(row["cell_id"]) for row in adapter_index["adapters"]}
    if indexed_states != set(initial_states + expansion_states):
        raise ValueError("adapter evidence index differs from v3 state set")
    if {row["state_id"] for row in overlap["target_manifests"]} != set(initial_states):
        raise ValueError("overlap audit differs from the initial state set")

    u0 = protocol["stage_u0a_fixed_state_reliability"]
    u1 = protocol["stage_u1_cross_state_transfer"]
    if u0.get("historical_measurements_in_primary") is not False:
        raise ValueError("v3 U0 primary must not reuse historical measurements")
    if 48 * len(u0["probe_seeds"]) != int(u0["new_measurements"]):
        raise ValueError("U0 measurement count is inconsistent")
    if 48 * len(initial_states) * len(u1["probe_seeds"]) != int(
        u1["initial_new_measurements"]
    ):
        raise ValueError("U1 initial measurement count is inconsistent")

    result = {
        "schema_version": "candidate-utility-state-dependence-preflight-v3",
        "status": "READY_FOR_GPU_QUALIFICATION",
        "gpu_accessed": False,
        "gpu_authorized": False,
        "protocol_sha256": file_sha256(protocol_path),
        "panel_sha256": file_sha256(panel_path),
        "overlap_audit_sha256": file_sha256(overlap_path),
        "adapter_index_sha256": file_sha256(adapter_index_path),
        "candidate_count": len(candidate_ids),
        "candidate_id_sha256": id_sha256(candidate_ids),
        "utility_record_count": len(utility_ids),
        "utility_record_id_sha256": id_sha256(utility_ids),
        "historical_measurements_reused": False,
        "planned_new_measurements": {
            "u0a_fixed_state_reliability": int(u0["new_measurements"]),
            "u1_initial_four_states_two_probe_seeds": int(
                u1["initial_new_measurements"]
            ),
            "u1_optional_expansion_four_states": int(u1["expansion_new_measurements"]),
        },
        "initial_adapter_states": initial_states,
        "expansion_adapter_states": expansion_states,
        "input_files": {
            "gsm8k_records": file_sha256(gsm8k_records_path),
            "tulu_pool": file_sha256(tulu_pool_path),
            "formal_token_audit": file_sha256(formal_token_audit_path),
        },
        "remaining_before_formal_gpu": [
            "run_full_CPU_tests_and_contract_only",
            "run_two_candidate_three_seed_GPU_qualification_after_explicit_authorization",
            "independently_audit_qualification",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument("--adapter-index", type=Path, required=True)
    parser.add_argument("--gsm8k-records", type=Path, required=True)
    parser.add_argument("--tulu-pool", type=Path, required=True)
    parser.add_argument("--formal-token-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(
        protocol_path=args.protocol.resolve(),
        panel_path=args.panel.resolve(),
        overlap_path=args.overlap_audit.resolve(),
        adapter_index_path=args.adapter_index.resolve(),
        gsm8k_records_path=args.gsm8k_records.resolve(),
        tulu_pool_path=args.tulu_pool.resolve(),
        formal_token_audit_path=args.formal_token_audit.resolve(),
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "historical_measurements_reused": result[
                    "historical_measurements_reused"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
