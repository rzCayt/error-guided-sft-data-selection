"""Immutable release and human-authorization gate for formal v8 cells."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from eg_sft.evaluation.phase2_v7_canary import file_sha256, read_json
from eg_sft.experiment.phase2_v7_environment import canonical_json_sha256
from eg_sft.experiment.phase2_v8_snapshot import validate_snapshot_manifest


PROTOCOL_ID = "phase2-clean-common24-v8"
HUMAN_CONFIRMATION = "START_PHASE2_V8_COMMON24"


def validate_semantic_code_manifest(*, repo_root: Path, manifest_path: Path) -> dict:
    root = repo_root.resolve()
    path = manifest_path.resolve()
    payload = read_json(path)
    if payload.get("schema_version") != "phase2-v8-semantic-code-manifest-v1":
        raise ValueError("unexpected v8 semantic manifest schema")
    content = {
        key: value
        for key, value in payload.items()
        if key != "manifest_content_sha256"
    }
    if payload.get("manifest_content_sha256") != canonical_json_sha256(content):
        raise ValueError("v8 semantic manifest content hash changed")
    files = payload.get("files", [])
    if not files:
        raise ValueError("v8 semantic manifest is empty")
    for row in files:
        source = (root / str(row["path"])).resolve()
        source.relative_to(root)
        if not source.is_file() or file_sha256(source) != row["sha256"]:
            raise ValueError(f"v8 semantic source changed: {row['path']}")
    return {
        "status": "PASS",
        "manifest_sha256": file_sha256(path),
        "file_count": len(files),
    }


def validate_deployment_tree(*, repo_root: Path, manifest_path: Path) -> dict:
    root = repo_root.resolve()
    path = manifest_path.resolve()
    payload = read_json(path)
    if payload.get("schema_version") != "phase2-v8-release-manifest-v1":
        raise ValueError("unexpected v8 deployment manifest schema")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected v8 deployment protocol")
    files = payload.get("files", [])
    if int(payload.get("file_count", -1)) != len(files) or not files:
        raise ValueError("v8 deployment manifest count changed")
    for row in files:
        source = (root / str(row["path"])).resolve()
        source.relative_to(root)
        if (
            not source.is_file()
            or source.stat().st_size != int(row["size_bytes"])
            or file_sha256(source) != row["sha256"]
        ):
            raise ValueError(f"v8 deployment file changed: {row['path']}")
    return {
        "status": "PASS",
        "manifest_sha256": file_sha256(path),
        "manifest_content_sha256": canonical_json_sha256(payload),
        "file_count": len(files),
    }


def require_clean_git(repo_root: Path) -> str:
    root = repo_root.resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("v8 formal execution requires a clean Git worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise ValueError("v8 Git commit is invalid")
    return head


def _exact_sha(payload: Mapping[str, Any], role: str, path: Path) -> None:
    expected = str(payload.get("bindings", {}).get(role, ""))
    if len(expected) != 64 or file_sha256(path.resolve()) != expected:
        raise ValueError(f"v8 release authorization binding changed: {role}")


def validate_release_authorization(
    *,
    repo_root: Path,
    release_go_path: Path,
    worker_id: str,
    canonical_runtime_path: Path,
    deployment_manifest_path: Path,
    release_archive_path: Path,
    environment_manifest_path: Path,
    backend_report_path: Path,
    training_anchor_report_path: Path,
    model_snapshot: Path,
    model_manifest_path: Path,
    tokenizer_manifest_path: Path,
) -> dict:
    root = repo_root.resolve()
    release = read_json(release_go_path.resolve())
    if (
        release.get("schema_version") != "phase2-v8-release-go-v2"
        or release.get("status") != "GO"
        or release.get("protocol_id") != PROTOCOL_ID
        or release.get("human_authorization") != HUMAN_CONFIRMATION
    ):
        raise ValueError("v8 RELEASE_GO is not human-authorized")
    actual_paths = {
        "canonical_runtime": canonical_runtime_path,
        "deployment_manifest": deployment_manifest_path,
        "release_archive": release_archive_path,
        "training_anchor_final": training_anchor_report_path,
        "model_tree_manifest": model_manifest_path,
        "tokenizer_tree_manifest": tokenizer_manifest_path,
    }
    for role, path in actual_paths.items():
        _exact_sha(release, role, path)
    workers = release.get("workers", {})
    worker = workers.get(worker_id, {})
    if worker.get("environment_manifest_sha256") != file_sha256(
        environment_manifest_path.resolve()
    ):
        raise ValueError("v8 release environment binding changed")
    if worker.get("backend_report_sha256") != file_sha256(
        backend_report_path.resolve()
    ):
        raise ValueError("v8 release backend binding changed")
    deployment = validate_deployment_tree(
        repo_root=root, manifest_path=deployment_manifest_path
    )
    model_manifest = read_json(model_manifest_path.resolve())
    tokenizer_manifest = read_json(tokenizer_manifest_path.resolve())
    model_tree_sha = validate_snapshot_manifest(
        snapshot=model_snapshot, manifest=model_manifest
    )
    tokenizer_tree_sha = validate_snapshot_manifest(
        snapshot=model_snapshot, manifest=tokenizer_manifest
    )
    git_commit = require_clean_git(root)
    if release.get("git_commit") != git_commit:
        raise ValueError("v8 release Git commit changed")
    return {
        "schema_version": "phase2-v8-cell-release-binding-v1",
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "worker_id": worker_id,
        "release_go_sha256": file_sha256(release_go_path.resolve()),
        "deployment_manifest_sha256": deployment["manifest_sha256"],
        "release_archive_sha256": file_sha256(release_archive_path.resolve()),
        "canonical_runtime_sha256": file_sha256(canonical_runtime_path.resolve()),
        "model_tree_content_sha256": model_tree_sha,
        "tokenizer_tree_content_sha256": tokenizer_tree_sha,
        "environment_manifest_sha256": file_sha256(environment_manifest_path.resolve()),
        "backend_report_sha256": file_sha256(backend_report_path.resolve()),
        "training_anchor_report_sha256": file_sha256(
            training_anchor_report_path.resolve()
        ),
        "git_commit": git_commit,
        "accuracy_withheld": True,
    }


def validate_release_gate_from_environment(*, repo_root: Path) -> dict:
    required = {
        "release_go_path": "EG_SFT_PHASE2_V8_RELEASE_GO",
        "worker_id": "EG_SFT_WORKER_ID",
        "canonical_runtime_path": "EG_SFT_PHASE2_V8_CANONICAL_RUNTIME",
        "deployment_manifest_path": "EG_SFT_PHASE2_V8_DEPLOYMENT_MANIFEST",
        "release_archive_path": "EG_SFT_PHASE2_V8_RELEASE_ARCHIVE",
        "environment_manifest_path": "EG_SFT_PHASE2_V8_ENVIRONMENT_MANIFEST",
        "backend_report_path": "EG_SFT_PHASE2_V8_BACKEND_REPORT",
        "training_anchor_report_path": "EG_SFT_PHASE2_V8_TRAINING_ANCHOR",
        "model_snapshot": "EG_SFT_FROZEN_MODEL_SNAPSHOT",
        "model_manifest_path": "EG_SFT_PHASE2_V8_MODEL_MANIFEST",
        "tokenizer_manifest_path": "EG_SFT_PHASE2_V8_TOKENIZER_MANIFEST",
    }
    values = {key: os.environ.get(env, "").strip() for key, env in required.items()}
    missing = [required[key] for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"v8 formal release environment is incomplete: {missing}")
    worker_id = values.pop("worker_id")
    return validate_release_authorization(
        repo_root=repo_root,
        worker_id=worker_id,
        **{key: Path(value) for key, value in values.items()},
    )
