#!/usr/bin/env python3
"""CPU-only independent audit for a completed state-dependence v3 probe run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from eg_sft.experiment.state_drift import validate_resume_rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def audit_run(
    *, run_dir: Path, protocol_path: Path, preflight_path: Path, panel_path: Path
) -> dict[str, Any]:
    contract = read_json(run_dir / "run_contract.json")
    complete = read_json(run_dir / "COMPLETE.json")
    rows = read_jsonl(run_dir / "utility_measurements.jsonl")
    if contract.get("schema_version") != "candidate-utility-state-probe-run-contract-v3":
        raise ValueError("run does not use the v3 contract")
    if contract.get("measurement_semantics") != "state_conditioned_local_utility_fresh_adamw":
        raise ValueError("measurement semantics changed")
    if contract.get("historical_measurements_reused") is not False:
        raise ValueError("v3 primary run reused historical measurements")
    if complete.get("schema_version") != "candidate-utility-state-probe-complete-v3":
        raise ValueError("run does not use the v3 completion schema")
    if complete.get("status") != "PASS":
        raise ValueError("COMPLETE status is not PASS")
    if int(complete.get("reused_measurement_count", -1)) != 0:
        raise ValueError("COMPLETE reports reused measurements")
    if contract["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("run protocol SHA differs from audited protocol")
    if contract["preflight_sha256"] != file_sha256(preflight_path):
        raise ValueError("run preflight SHA differs from audited preflight")
    if contract["panel_sha256"] != file_sha256(panel_path):
        raise ValueError("run panel SHA differs from audited panel")

    completed_keys = validate_resume_rows(plan=contract["plan"], rows=rows)
    if len(completed_keys) != int(contract["plan"]["new_measurement_count"]):
        raise ValueError("measurement rows do not cover the frozen plan")
    if int(contract["plan"].get("reused_measurement_count", -1)) != 0:
        raise ValueError("frozen plan contains reused measurements")
    if int(complete["new_measurement_count"]) != len(rows):
        raise ValueError("COMPLETE measurement count differs from JSONL")
    measurement_sha = file_sha256(run_dir / "utility_measurements.jsonl")
    if complete["measurement_sha256"] != measurement_sha:
        raise ValueError("measurement JSONL SHA differs from COMPLETE")

    snapshot_sha = str(complete["state_snapshot_sha256"])
    if len(snapshot_sha) != 64:
        raise ValueError("invalid state snapshot SHA")
    numeric_fields = (
        "state_utility_loss",
        "post_utility_loss",
        "utility",
        "candidate_train_loss",
        "gradient_norm_before_clipping",
        "restore_probe_loss_difference",
    )
    for row in rows:
        if any(not math.isfinite(float(row[field])) for field in numeric_fields):
            raise ValueError("measurement contains a non-finite numeric value")
        if float(row["restore_probe_loss_difference"]) > 1e-7:
            raise ValueError("adapter restore probe-loss difference exceeds tolerance")
        if int(row["trainable_parameters"]) != 18464768:
            raise ValueError("trainable parameter count changed")
        if row.get("state_snapshot_sha256") != snapshot_sha:
            raise ValueError("measurement state snapshot SHA changed within the run")
        if row.get("zero_adapter_initialization_seed") != contract.get(
            "zero_adapter_initialization_seed"
        ):
            raise ValueError("zero-state initialization seed changed within the run")

    result = {
        "schema_version": "candidate-utility-state-probe-audit-v3",
        "status": "PASS",
        "state_id": contract["state_id"],
        "measurement_count": len(rows),
        "probe_seeds": contract["probe_seeds"],
        "candidate_count": len(contract["candidate_ids"]),
        "historical_measurements_reused": False,
        "state_snapshot_sha256": snapshot_sha,
        "measurement_sha256": measurement_sha,
        "contract_sha256": file_sha256(run_dir / "run_contract.json"),
        "complete_sha256": file_sha256(run_dir / "COMPLETE.json"),
        "claim_boundary": (
            "This audit validates state-conditioned local one-step probes with a fresh "
            "optimizer. It does not establish optimizer-trajectory influence or explain "
            "the full downstream result."
        ),
    }
    output = run_dir / "INDEPENDENT_AUDIT.json"
    if output.exists():
        if read_json(output) != result:
            raise ValueError("existing independent audit differs from recomputation")
    else:
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(
        run_dir=args.run_dir.resolve(),
        protocol_path=args.protocol.resolve(),
        preflight_path=args.preflight.resolve(),
        panel_path=args.panel.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "measurement_count": result["measurement_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
