"""Accuracy-blind progress, ETA, cost and unblinding gate for v8."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v7_control import (  # noqa: E402
    Phase2StateStore,
    validate_complete_evidence,
    worker_schedule,
)


def _hours(events: list[dict]) -> float:
    attempts = {}
    for row in events:
        attempt = row.get("attempt_id")
        target = row.get("target_state")
        if attempt and target in {"LOCKED", "COMPLETE", "FAILED"}:
            attempts.setdefault(attempt, {})[target] = datetime.fromisoformat(
                row["recorded_at_utc"]
            )
    seconds = 0.0
    for states in attempts.values():
        end = states.get("COMPLETE") or states.get("FAILED")
        if end is not None and "LOCKED" in states:
            seconds += max(0.0, (end - states["LOCKED"]).total_seconds())
    return seconds / 3600


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_clean_common24_v8_canonical.json")
    )
    for worker in ("gpu0", "gpu1"):
        parser.add_argument(f"--control-{worker}", type=Path, required=True)
        parser.add_argument(f"--packages-{worker}", type=Path, required=True)
    parser.add_argument("--gpu-hour-rate", type=float, default=1.88)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stores = {
        worker: Phase2StateStore(
            root=getattr(args, f"control_{worker}"), matrix_path=args.config.resolve()
        )
        for worker in ("gpu0", "gpu1")
    }
    package_roots = {
        worker: getattr(args, f"packages_{worker}").resolve()
        for worker in ("gpu0", "gpu1")
    }
    rows = []
    completed_durations = []
    total_hours = 0.0
    for worker, store in stores.items():
        for position, cell in enumerate(worker_schedule(store.matrix, worker), start=1):
            state = store.read_state(cell)
            elapsed = _hours(read_jsonl(store.cell_dir(cell) / "events.jsonl"))
            total_hours += elapsed
            if state["state"] == "COMPLETE":
                validate_complete_evidence(
                    state=state,
                    required_hashes=("formal_audit_sha256", "ood_audit_sha256", "evidence_package_sha256"),
                )
                package = package_roots[worker] / state["evidence"]["evidence_package_name"]
                if not package.is_file() or file_sha256(package) != state["evidence"]["evidence_package_sha256"]:
                    raise ValueError("v8 COMPLETE evidence package changed")
                completed_durations.append(elapsed)
            rows.append({"worker_id": worker, "worker_position": position, "state": state["state"], "attempt_count": sum(event.get("target_state") == "LOCKED" for event in read_jsonl(store.cell_dir(cell) / "events.jsonl")), "recorded_gpu_hours": elapsed})
    complete = sum(row["state"] == "COMPLETE" for row in rows)
    failed = sum(row["state"] == "FAILED" for row in rows)
    remaining = 24 - complete
    median = statistics.median(completed_durations) if completed_durations else 2.824
    p90 = sorted(completed_durations)[max(0, int(len(completed_durations) * 0.9) - 1)] if completed_durations else 3.393
    payload = {
        "schema_version": "phase2-v8-blind-progress-v1",
        "status": "AUDITED_PASS" if complete == 24 and failed == 0 else "IN_PROGRESS",
        "complete_count": complete,
        "required_count": 24,
        "failed_count": failed,
        "remaining_count": remaining,
        "ra_outreach_minimum_reached": complete >= 4,
        "ra_outreach_preferred_reached": complete >= 8,
        "recorded_gpu_hours": total_hours,
        "estimated_compute_cost_cny": total_hours * args.gpu_hour_rate,
        "estimated_remaining_dual_gpu_wall_hours_median": ((remaining + 1) // 2) * median,
        "estimated_remaining_dual_gpu_wall_hours_p90": ((remaining + 1) // 2) * p90,
        "unblinding_permitted": complete == 24 and failed == 0,
        "automatic_unblinding": False,
        "historical_seed17_in_primary": False,
        "free_mix_in_primary": False,
        "accuracy_withheld": True,
        "actual_method_names_withheld": True,
        "engineering_rows": rows,
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
