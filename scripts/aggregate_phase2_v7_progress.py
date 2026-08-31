"""CPU-only engineering progress, ETA, cost and 48-cell unblinding gate."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    read_jsonl,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v7_control import (  # noqa: E402
    Phase2StateStore,
    validate_complete_evidence,
    worker_schedule,
)


def _elapsed_hours(events: list[dict[str, Any]]) -> float:
    by_attempt: dict[str, dict[str, datetime]] = {}
    for event in events:
        attempt = event.get("attempt_id")
        if not attempt:
            continue
        target = str(event.get("target_state", ""))
        if target not in {"LOCKED", "COMPLETE", "FAILED"}:
            continue
        by_attempt.setdefault(str(attempt), {})[target] = datetime.fromisoformat(
            str(event["recorded_at_utc"])
        )
    seconds = 0.0
    for states in by_attempt.values():
        if "LOCKED" not in states:
            continue
        end = states.get("COMPLETE") or states.get("FAILED")
        if end is not None:
            seconds += max(0.0, (end - states["LOCKED"]).total_seconds())
    return seconds / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_crossed_48cell_v7.json"),
    )
    parser.add_argument("--control-gpu0", type=Path, required=True)
    parser.add_argument("--control-gpu1", type=Path, required=True)
    parser.add_argument("--packages-gpu0", type=Path, required=True)
    parser.add_argument("--packages-gpu1", type=Path, required=True)
    parser.add_argument("--parent-gate", type=Path, required=True)
    parser.add_argument("--gpu-hour-rate", type=float, default=1.88)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    stores = {
        "gpu0": Phase2StateStore(root=args.control_gpu0, matrix_path=config_path),
        "gpu1": Phase2StateStore(root=args.control_gpu1, matrix_path=config_path),
    }
    parent_gate = read_json(args.parent_gate.resolve())
    parent_ready = (
        int(parent_gate.get("job_count", -1)) == 16
        and int(parent_gate.get("formal_audited_pass_count", -1)) == 16
        and int(parent_gate.get("ood_audited_pass_count", -1)) == 16
        and parent_gate.get("unblinding_permitted") is True
    )
    package_roots = {
        "gpu0": args.packages_gpu0.resolve(),
        "gpu1": args.packages_gpu1.resolve(),
    }
    engineering_rows = []
    duration_hours = []
    accumulated_hours = 0.0
    for worker_id, store in stores.items():
        for position, cell_id in enumerate(
            worker_schedule(store.matrix, worker_id), start=1
        ):
            state = store.read_state(cell_id)
            events_path = store.cell_dir(cell_id) / "events.jsonl"
            hours = _elapsed_hours(read_jsonl(events_path))
            accumulated_hours += hours
            if state["state"] == "COMPLETE":
                validate_complete_evidence(
                    state=state,
                    required_hashes=(
                        "formal_audit_sha256",
                        "ood_audit_sha256",
                        "evidence_package_sha256",
                    ),
                )
                package = package_roots[worker_id] / str(
                    state["evidence"]["evidence_package_name"]
                )
                if (
                    not package.is_file()
                    or file_sha256(package)
                    != state["evidence"]["evidence_package_sha256"]
                ):
                    raise ValueError(f"completed evidence package changed: {worker_id}/{position}")
                duration_hours.append(hours)
            engineering_rows.append(
                {
                    "worker_id": worker_id,
                    "worker_position": position,
                    "state": state["state"],
                    "attempt_count": sum(
                        event.get("target_state") == "LOCKED"
                        for event in read_jsonl(events_path)
                    ),
                    "recorded_gpu_hours": hours,
                }
            )
    complete_count = sum(row["state"] == "COMPLETE" for row in engineering_rows)
    failed_count = sum(row["state"] == "FAILED" for row in engineering_rows)
    remaining = 32 - complete_count
    median_hours = statistics.median(duration_hours) if duration_hours else 2.824
    p90_hours = (
        sorted(duration_hours)[max(0, int(len(duration_hours) * 0.9) - 1)]
        if duration_hours
        else 3.393
    )
    estimated_wall_median = ((remaining + 1) // 2) * median_hours
    estimated_wall_p90 = ((remaining + 1) // 2) * p90_hours
    new_ready = complete_count == 32 and failed_count == 0
    payload = {
        "schema_version": "phase2-v7-blind-progress-v1",
        "status": "AUDITED_PASS" if new_ready else "IN_PROGRESS",
        "parent_16_audited_pass": parent_ready,
        "new_complete_count": complete_count,
        "new_required_count": 32,
        "total_complete_after_merge": 16 + complete_count if parent_ready else complete_count,
        "failed_count": failed_count,
        "remaining_count": remaining,
        "recorded_gpu_hours": accumulated_hours,
        "estimated_compute_cost_cny": accumulated_hours * args.gpu_hour_rate,
        "estimated_remaining_dual_gpu_wall_hours_median": estimated_wall_median,
        "estimated_remaining_dual_gpu_wall_hours_p90": estimated_wall_p90,
        "unblinding_permitted": parent_ready and new_ready,
        "automatic_unblinding": False,
        "accuracy_withheld": True,
        "actual_method_names_withheld": True,
        "engineering_rows": engineering_rows,
        "claim_boundary": (
            "Engineering integrity gate only; contains no accuracy or method comparison."
        ),
    }
    output = args.output.resolve()
    write_exclusive_or_verify(output, canonical_json_bytes(payload))
    write_exclusive_or_verify(
        output.with_suffix(output.suffix + ".sha256"),
        f"{file_sha256(output)}  {output.name}\n".encode("ascii"),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
