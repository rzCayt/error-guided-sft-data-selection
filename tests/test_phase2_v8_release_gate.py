from __future__ import annotations

import json
from pathlib import Path

import pytest

from eg_sft.experiment.phase2_v8_release_gate import (
    validate_deployment_tree,
    validate_semantic_code_manifest,
)
from eg_sft.experiment.phase2_v7_environment import canonical_json_sha256
from eg_sft.training.b500 import file_sha256


def test_semantic_manifest_rehashes_every_source(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    content = {
        "schema_version": "phase2-v8-semantic-code-manifest-v1",
        "protocol_id": "phase2-clean-common24-v8",
        "files": [{"path": "source.py", "sha256": file_sha256(source)}],
    }
    manifest = tmp_path / "semantic.json"
    manifest.write_text(
        json.dumps(content | {"manifest_content_sha256": canonical_json_sha256(content)}),
        encoding="utf-8",
    )
    assert validate_semantic_code_manifest(
        repo_root=tmp_path, manifest_path=manifest
    )["status"] == "PASS"
    source.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantic source changed"):
        validate_semantic_code_manifest(repo_root=tmp_path, manifest_path=manifest)


def test_deployment_manifest_rehashes_every_file(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("a", encoding="utf-8")
    payload = {
        "schema_version": "phase2-v8-release-manifest-v1",
        "protocol_id": "phase2-clean-common24-v8",
        "file_count": 1,
        "files": [
            {"path": "a.txt", "size_bytes": 1, "sha256": file_sha256(source)}
        ],
    }
    manifest = tmp_path / "DEPLOYMENT_MANIFEST.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_deployment_tree(
        repo_root=tmp_path, manifest_path=manifest
    )["status"] == "PASS"
    source.write_text("b", encoding="utf-8")
    with pytest.raises(ValueError, match="deployment file changed"):
        validate_deployment_tree(repo_root=tmp_path, manifest_path=manifest)
