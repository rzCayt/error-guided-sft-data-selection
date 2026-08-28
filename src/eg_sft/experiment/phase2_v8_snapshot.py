"""Fail-closed local snapshot controls for every Phase-2 v8 GPU path."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eg_sft.experiment.phase2_v7_environment import (
    canonical_json_sha256,
    file_sha256,
)


SNAPSHOT_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
TOKENIZER_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def file_subset_manifest(root: Path, *, relative_names: Sequence[str] | None = None) -> dict:
    root = root.resolve(strict=True)
    allowed = None if relative_names is None else set(relative_names)
    files = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".locks/") or (
            allowed is not None and relative not in allowed
        ):
            continue
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise ValueError("v8 snapshot file manifest is empty")
    content = {"schema_version": "phase2-v8-file-tree-v1", "files": files}
    return content | {"manifest_content_sha256": canonical_json_sha256(content)}


def tokenizer_file_names(snapshot: Path) -> list[str]:
    names = sorted(
        path.relative_to(snapshot).as_posix()
        for path in snapshot.iterdir()
        if path.is_file() and path.name in TOKENIZER_NAMES
    )
    required = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    if not required.issubset(names):
        raise ValueError("v8 local snapshot tokenizer files are incomplete")
    return names


def validate_snapshot_manifest(*, snapshot: Path, manifest: Mapping[str, Any]) -> str:
    if manifest.get("schema_version") != "phase2-v8-file-tree-v1":
        raise ValueError("unexpected v8 snapshot manifest schema")
    content = {"schema_version": manifest["schema_version"], "files": manifest["files"]}
    if manifest.get("manifest_content_sha256") != canonical_json_sha256(content):
        raise ValueError("v8 snapshot manifest content hash changed")
    snapshot = snapshot.resolve(strict=True)
    for row in manifest["files"]:
        path = (snapshot / str(row["path"])).resolve()
        path.relative_to(snapshot)
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size"])
            or file_sha256(path) != row["sha256"]
        ):
            raise ValueError(f"v8 snapshot file changed: {row['path']}")
    return canonical_json_sha256(content)


def configure_frozen_snapshot(snapshot: Path) -> Path:
    snapshot = snapshot.resolve(strict=True)
    if snapshot.name != SNAPSHOT_REVISION:
        raise ValueError("v8 frozen model snapshot revision changed")
    os.environ.update(
        {
            "EG_SFT_FROZEN_MODEL_SNAPSHOT": str(snapshot),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    return snapshot


def frozen_model_source(model_config: Mapping[str, Any]) -> tuple[str | Path, dict[str, Any]]:
    raw = os.environ.get("EG_SFT_FROZEN_MODEL_SNAPSHOT", "").strip()
    if raw:
        snapshot = configure_frozen_snapshot(Path(raw))
        if str(model_config.get("revision")) != SNAPSHOT_REVISION:
            raise ValueError("v8 protocol model revision changed")
        return snapshot, {"local_files_only": True}
    return str(model_config["repo_id"]), {"revision": model_config["revision"]}


def current_single_gpu_identity() -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("v8 requires exactly one physical GPU per worker instance")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 3:
        raise ValueError("unexpected v8 GPU identity output")
    return {"sku": values[0], "uuid": values[1], "driver_version": values[2]}
