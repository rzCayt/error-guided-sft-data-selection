"""Create non-overwriting run directories with bounded provenance metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def create_run_manifest(
    *,
    output_root: Path,
    repo_root: Path,
    stage: str,
    config: dict[str, Any],
    seed: int,
    command: list[str],
    dataset_revisions: dict[str, str],
    model_revision: str | None,
    extra: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create a new run directory and write ``manifest.json`` exactly once."""

    if not stage or any(character in stage for character in r'\/:*?"<>|'):
        raise ValueError("stage must be a non-empty filesystem-safe name")

    timestamp = (now or datetime.now(UTC)).astimezone(UTC)
    config_hash = stable_config_hash(config)
    run_id = f"{timestamp:%Y%m%dT%H%M%SZ}_{stage}_{config_hash[:10]}_s{seed}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "started_at_utc": timestamp.isoformat(),
        "git_commit": _git_commit(repo_root),
        "config_hash": config_hash,
        "config": config,
        "seed": seed,
        "command": command,
        "dataset_revisions": dataset_revisions,
        "model_revision": model_revision,
        "python": sys.version,
        "platform": platform.platform(),
    }
    if extra:
        manifest["extra"] = extra

    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return run_dir, manifest
