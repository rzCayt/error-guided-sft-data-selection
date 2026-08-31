"""Read and hash-audit Phase-1/Phase-2 cell evidence archives without extraction."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import PurePosixPath, Path
from typing import Any

from eg_sft.evaluation.phase2_v7_canary import file_sha256


TASK_MEMBERS = {
    "gsm8k": "evaluation/merged/raw_outputs.jsonl",
    "svamp": "evaluation/ood/svamp/merged/raw_outputs.jsonl",
    "asdiv_numeric": "evaluation/ood/asdiv_numeric/merged/raw_outputs.jsonl",
    "multiarith": "evaluation/ood/multiarith/merged/raw_outputs.jsonl",
}


def _json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{index} must contain a JSON object")
        rows.append(value)
    return rows


def _safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe evidence member path: {name}")


def read_cell_evidence_archive(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"evidence archive is missing: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="ascii").split()[0]
        if file_sha256(path) != expected:
            raise ValueError(f"evidence archive sidecar changed: {path.name}")
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for name in members:
            _safe_member_name(name)

        def payload(name: str) -> bytes:
            if name not in members:
                raise ValueError(f"evidence archive member is missing: {name}")
            handle = archive.extractfile(members[name])
            if handle is None:
                raise ValueError(f"cannot read evidence member: {name}")
            return handle.read()

        evidence_manifest = _json_bytes(
            payload("EVIDENCE_MANIFEST.json"), "EVIDENCE_MANIFEST.json"
        )
        listed = evidence_manifest.get("files")
        if not isinstance(listed, list) or not listed:
            raise ValueError("evidence manifest file list is empty")
        for row in listed:
            name = str(row["archive_path"])
            _safe_member_name(name)
            content = payload(name)
            if (
                len(content) != int(row["size_bytes"])
                or hashlib.sha256(content).hexdigest() != row["sha256"]
            ):
                raise ValueError(f"evidence member hash changed: {name}")
        run_manifest = _json_bytes(payload("manifest.json"), "manifest.json")
        formal = _json_bytes(
            payload("audit/formal_cell_audit.json"), "formal audit"
        )
        ood = _json_bytes(payload("audit/ood_audit.json"), "OOD audit")
        if formal.get("status") != "PASS" or ood.get("status") != "PASS":
            raise ValueError("evidence archive contains a failed audit")
        config = run_manifest.get("config", {})
        cell_id = str(config.get("cell_id", ""))
        if (
            not cell_id
            or formal.get("cell_id") != cell_id
            or evidence_manifest.get("cell_id") != cell_id
        ):
            raise ValueError("evidence cell identity changed")
        tasks = {
            task: _jsonl_bytes(payload(member), member)
            for task, member in TASK_MEMBERS.items()
        }
        training = _json_bytes(
            payload("training_complete/training_metrics.json"), "training metrics"
        )
        cell = {
            "cell_id": cell_id,
            "method": str(config["method"]),
            "replicate_index": int(config["replicate_index"]),
            "train_seed": int(run_manifest["seed"]),
            "tasks": tasks,
            "training": training,
        }
        evidence = {
            "cell_id": cell_id,
            "archive_name": path.name,
            "archive_sha256": file_sha256(path),
            "run_id": run_manifest["run_id"],
            "formal_audit_sha256": hashlib.sha256(
                payload("audit/formal_cell_audit.json")
            ).hexdigest(),
            "ood_audit_sha256": hashlib.sha256(
                payload("audit/ood_audit.json")
            ).hexdigest(),
            "task_raw_output_sha256": {
                task: hashlib.sha256(payload(member)).hexdigest()
                for task, member in TASK_MEMBERS.items()
            },
        }
        return cell, evidence


def load_evidence_roots(roots: list[Path]) -> tuple[list[dict], list[dict]]:
    archives = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"evidence root is missing: {root}")
        archives.extend(sorted(root.glob("*_evidence.tar.gz")))
    cells = []
    evidence = []
    for archive in archives:
        cell, row = read_cell_evidence_archive(archive)
        cells.append(cell)
        evidence.append(row)
    ids = [str(row["cell_id"]) for row in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("evidence roots contain duplicate cells")
    return cells, evidence
