"""Single-authority runtime manifest for the v8 deployment package."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eg_sft.evaluation.phase2_v7_canary import file_sha256, read_json
from eg_sft.experiment.phase2_v8_release_gate import validate_semantic_code_manifest


SCHEMA_VERSION = "phase2-v8-canonical-runtime-files-v1"
CANONICAL_RUNTIME_RELATIVE = "configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json"
EXPECTED_ROLES = {
    "primary_matrix",
    "statistical_protocol",
    "training_anchor_protocol",
    "canary_contract",
    "stop_go_rules",
    "parent_matrix",
    "base_recipe",
    "research_protocol",
    "information_gates",
    "semantic_code_manifest",
    "materialized_contracts",
    "materialized_contract_audit",
}


def validate_canonical_runtime(
    *, repo_root: Path, manifest_path: Path
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest_path.relative_to(repo_root)
    expected_manifest_path = (repo_root / CANONICAL_RUNTIME_RELATIVE).resolve()
    if manifest_path != expected_manifest_path:
        raise ValueError("v8 runtime attempted a non-release canonical authority")
    payload = read_json(manifest_path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected v8 canonical runtime schema")
    if payload.get("protocol_id") != "phase2-clean-common24-v8":
        raise ValueError("unexpected v8 canonical protocol ID")
    if payload.get("status") != "FROZEN":
        raise ValueError("v8 canonical runtime is not frozen")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("v8 canonical runtime file list is empty")
    roles = {}
    paths = set()
    for row in files:
        role = str(row.get("role", ""))
        relative = str(row.get("path", "")).replace("\\", "/")
        if not role or not relative or role in roles or relative in paths:
            raise ValueError("v8 canonical role/path is missing or duplicated")
        path = (repo_root / relative).resolve()
        path.relative_to(repo_root)
        if not path.is_file() or file_sha256(path) != row.get("sha256"):
            raise ValueError(f"v8 canonical runtime file changed: {role}")
        roles[role] = {"path": path, "sha256": row["sha256"]}
        paths.add(relative)
    observed_roles = set(roles)
    if observed_roles != EXPECTED_ROLES:
        missing = sorted(EXPECTED_ROLES - observed_roles)
        unexpected = sorted(observed_roles - EXPECTED_ROLES)
        raise ValueError(
            f"v8 canonical runtime roles changed: missing={missing}, unexpected={unexpected}"
        )
    if roles["primary_matrix"]["path"] != (repo_root / "configs/phase2_clean_common24_v8_canonical.json").resolve():
        raise ValueError("v8 canonical primary matrix path changed")
    semantic = validate_semantic_code_manifest(
        repo_root=repo_root,
        manifest_path=roles["semantic_code_manifest"]["path"],
    )
    return {
        "manifest": payload,
        "roles": roles,
        "manifest_sha256": file_sha256(manifest_path),
        "semantic_validation": semantic,
    }


def require_canonical_role(
    *, canonical: Mapping[str, Any], role: str, actual_path: Path
) -> None:
    roles = canonical.get("roles", {})
    if role not in roles or roles[role]["path"] != actual_path.resolve():
        raise ValueError(f"runtime attempted a noncanonical file for role: {role}")
