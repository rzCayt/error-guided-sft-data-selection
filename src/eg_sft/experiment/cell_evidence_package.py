"""Build a compact, auditable evidence archive for one completed Phase 1 cell."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "budget-equivalent-cell-evidence-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_files(run_dir: Path) -> list[Path]:
    """Return stable evidence files while excluding bulky resumable model states."""

    required = [
        "manifest.json",
        "resolved_recipe.json",
        "optimizer_step_tokens.jsonl",
        "runtime_events.jsonl",
        "cell_complete.json",
        "audit/formal_cell_audit.json",
        "audit/formal_cell_audit.sha256",
        "audit/ood_audit.json",
        "audit/ood_audit.sha256",
        "training_complete/training_metrics.json",
        "training_complete/token_audit.json",
        "training_complete/token_budget_audit.json",
        "training_complete/adapter/adapter_model.safetensors",
        "training_complete/adapter/adapter_config.json",
    ]
    missing = [relative for relative in required if not (run_dir / relative).is_file()]
    if missing:
        raise ValueError(f"completed cell is missing required evidence: {missing}")

    selected: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if relative.parts and relative.parts[0] == "checkpoints" and path.suffix == ".pt":
            continue
        selected.append(path)
    return sorted(selected, key=lambda path: path.relative_to(run_dir).as_posix())


def build_manifest(
    *, run_dir: Path, paths: Iterable[Path], extra_logs: Iterable[Path] = ()
) -> dict[str, Any]:
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    files = [
        {
            "archive_path": path.relative_to(run_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    log_rows = [
        {
            "archive_path": f"external_logs/{path.name}",
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(extra_logs, key=lambda item: item.name)
        if path.is_file()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run_id": run_manifest["run_id"],
        "cell_id": run_manifest["config"]["cell_id"],
        "run_git_commit": run_manifest["git_commit"],
        "config_hash": run_manifest["config_hash"],
        "accuracy_withheld": True,
        "checkpoint_tensor_files_omitted": True,
        "remote_source_paths_withheld": True,
        "files": files + log_rows,
        "file_count": len(files) + len(log_rows),
    }


def package_cell_evidence(
    *,
    run_dir: Path,
    output: Path,
    extra_logs: Iterable[Path] = (),
) -> tuple[dict[str, Any], str]:
    run_dir = run_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = evidence_files(run_dir)
    logs = [path.resolve() for path in extra_logs if path.is_file()]
    manifest = build_manifest(run_dir=run_dir, paths=paths, extra_logs=logs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="x:gz", compresslevel=6) as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(run_dir).as_posix(), recursive=False)
        for path in logs:
            archive.add(path, arcname=f"external_logs/{path.name}", recursive=False)
        payload = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        info = tarfile.TarInfo("EVIDENCE_MANIFEST.json")
        info.size = len(payload)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))
    archive_sha256 = file_sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    sidecar.write_text(f"{archive_sha256}  {output.name}\n", encoding="ascii")
    return manifest, archive_sha256
