"""Research status and compute admission, separate from the frozen 24-cell study."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from eg_sft.experiment.run_manifest import stable_config_hash


PHASE_CAPS = {"weeks_1_3": Decimal(2000), "weeks_4_8": Decimal(3500),
              "weeks_9_12": Decimal(1500), "reserve": Decimal(500)}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def source_identity(root: Path) -> dict[str, Any]:
    """Require a clean Git checkout or verify every member of a deployed source bundle."""
    bundle_path = root / "BUNDLE.json"
    if bundle_path.is_file():
        bundle = read_object(bundle_path)
        if bundle["schema_version"] != "autodl-source-bundle-v1":
            raise ValueError("Unknown deployment bundle schema")
        names = [entry["path"] for entry in bundle["files"]]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate bundle members")
        for entry in bundle["files"]:
            path = (root / entry["path"]).resolve()
            relative = path.relative_to(root.resolve())
            if any(part in {".git", "work"} for part in relative.parts):
                raise ValueError("Private files declared in source bundle")
            if path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Bundle source byte count changed: {entry['path']}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Bundle source changed: {entry['path']}")
        return {"kind": "verified_source_bundle", "git_commit": bundle["git_commit"],
                "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                            cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    if status:
        raise RuntimeError("Commit source changes before starting a GPU feasibility run")
    return {"kind": "clean_git_checkout", "git_commit": commit}


def amount(value: Any, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("Amount must be a number or decimal string")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError("Amount must be finite and non-negative (positive where required)")
    return result


def validate_budget(ledger: dict[str, Any]) -> dict[str, Decimal]:
    if ledger["schema_version"] != "research-upgrade-budget-v1" or ledger["currency"] != "CNY":
        raise ValueError("Unsupported budget schema or currency")
    if amount(ledger["total_cap_cny"]) != Decimal(7500):
        raise ValueError("Authorized aggregate cap is CNY 7500")
    caps = {key: amount(value) for key, value in ledger["phase_caps_cny"].items()}
    if caps != PHASE_CAPS:
        raise ValueError("Phase caps differ from the authorized plan")
    used = dict.fromkeys(caps, Decimal(0))
    batch_ids: set[str] = set()
    for entry in ledger["entries"]:
        batch_id = entry["batch_id"]
        if not isinstance(batch_id, str) or not batch_id or batch_id in batch_ids:
            raise ValueError("Empty or duplicate budget batch ID")
        batch_ids.add(batch_id)
        phase = entry["phase"]
        if phase not in caps or entry["status"] not in {"reserved", "settled", "cancelled"}:
            raise ValueError("Unknown budget phase or entry status")
        charge = amount(entry["charged_cny"])
        reservation = amount(entry["reserved_cny"])
        if entry["status"] == "reserved" and charge != 0:
            raise ValueError("Reserved entries must not also count a charge")
        if entry["status"] != "reserved" and reservation != 0:
            raise ValueError("Release reservation on settlement or cancellation")
        if entry["status"] == "cancelled" and charge != 0:
            raise ValueError("Cancelled entries must have zero charge")
        used[phase] += charge + reservation
    if any(used[phase] > caps[phase] for phase in caps):
        raise ValueError("Existing ledger exceeds a phase cap")
    if sum(used.values()) > amount(ledger["total_cap_cny"]):
        raise ValueError("Existing ledger exceeds the aggregate cap")
    return used


def plan_batch(
    *, ledger: dict[str, Any], profile: dict[str, Any], phase: str, batch_id: str,
    units: int, hourly_cny: str, quote_at: str, setup_hours: str = "0",
    storage_cny: str = "0", margin: str = "1.25",
) -> dict[str, Any]:
    """Admission estimate; an operator must reserve it before launching one batch."""
    used = validate_budget(ledger)
    if phase not in PHASE_CAPS or not batch_id or units <= 0 or isinstance(units, bool):
        raise ValueError("Invalid phase, batch ID or work-unit count")
    if any(entry["batch_id"] == batch_id for entry in ledger["entries"]):
        raise ValueError("Batch ID already exists in ledger")
    if profile["schema_version"] != "compute-profile-v1" or profile["status"] != "measured":
        raise ValueError("A measured compute profile is required; estimates cannot pose as timings")
    if not profile["gpu_model"] or not profile["work_unit"]:
        raise ValueError("GPU and work-unit identities are required")
    for field in ("measurement_sha256", "config_sha256"):
        if not isinstance(profile[field], str) or not SHA256.fullmatch(profile[field]):
            raise ValueError(f"Invalid profile evidence identity: {field}")
    measured = datetime.fromisoformat(profile["measured_at"])
    quote = datetime.fromisoformat(quote_at)
    if measured.tzinfo is None or quote.tzinfo is None:
        raise ValueError("Measurement and quote timestamps must include a timezone")
    seconds = amount(profile["wall_seconds_per_unit"], positive=True)
    rate = amount(hourly_cny, positive=True)
    factor = amount(margin, positive=True)
    if factor < 1:
        raise ValueError("Runtime margin must be at least one")
    hours = (seconds * units / 3600 + amount(setup_hours)) * factor
    cost = (hours * rate + amount(storage_cny)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if cost > PHASE_CAPS[phase] - used[phase] or cost > Decimal(7500) - sum(used.values()):
        raise ValueError(f"Batch exceeds remaining authorized budget: CNY {cost}")
    return {
        "schema_version": "compute-admission-v1", "status": "estimated_within_cap",
        "batch_id": batch_id, "phase": phase, "units": units,
        "gpu_model": profile["gpu_model"], "work_unit": profile["work_unit"],
        "quote_at": quote_at, "hourly_cny": str(rate), "storage_cny": str(amount(storage_cny)),
        "max_billed_hours": str(hours), "reserve_cny": str(cost),
        "ledger_sha256": stable_config_hash(ledger), "profile": profile,
        "launch_authorized": False,
        "remaining_actions": ["Recheck live quote", "Reserve cost in the single ledger",
                              "Set an instance on-time limit, including downloads and idle time",
                              "Stop the instance separately after preserving outputs"],
    }


def validate_registry(registry: dict[str, Any], root: Path) -> None:
    if registry["schema_version"] != "research-upgrade-registry-v1":
        raise ValueError("Unsupported new-study registry")
    rows = registry["studies"]
    ids = [row["study_id"] for row in rows]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("Study IDs must be unique and non-empty")
    pilots = [row for row in rows if row["kind"] == "research_pilot"]
    if len(pilots) != 3 or registry["maximum_research_pilots"] != 3:
        raise ValueError("The approved portfolio contains exactly three candidate pilots")
    for row in rows:
        if row["mode"] not in {"feasibility", "exploratory", "confirmatory", "replication"}:
            raise ValueError("Unknown research mode")
        if row["status"] not in {"planned", "blocked_input", "running", "completed", "stopped"}:
            raise ValueError("Unknown study status")
        if any(dependency not in ids for dependency in row["depends_on"]):
            raise ValueError("Unknown study prerequisite")
        for binding in row["frozen_inputs"]:
            path = (root / binding["path"]).resolve()
            path.relative_to(root.resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
                raise ValueError(f"Frozen input changed: {binding['path']}")
        artifacts = row["result_artifacts"]
        if row["status"] == "completed" and not artifacts:
            raise ValueError("A completed study requires actual result evidence")
        if row["status"] in {"planned", "blocked_input"} and artifacts:
            raise ValueError("An unrun study cannot contain results")
        for binding in artifacts:
            path = (root / binding["path"]).resolve()
            path.relative_to(root.resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
                raise ValueError("Research result identity mismatch")
        if row["gpu_result_available"] and not artifacts:
            raise ValueError("GPU result flag requires evidence")
