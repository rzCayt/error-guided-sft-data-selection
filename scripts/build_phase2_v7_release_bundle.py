"""Build ASCII-only, secret-scanned deployment and review archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


WINDOWS_PRIVATE_PATH = re.compile(rb"[A-Za-z]:\\(?:Users|RA)[^\r\n\"']*")
SSH_ENDPOINT = re.compile(rb"(?:ssh\s+-p\s+\d+|root@connect\.[^\s]+)", re.I)
TOKEN_VALUE = re.compile(
    rb"(?:password|api[_-]?key|access[_-]?token|secret)[ \t]*[:=][ \t]*[\"'][A-Za-z0-9+/_.-]{12,}[\"']|(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{12,}",
    re.I,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_ascii(relative: str) -> None:
    relative.encode("ascii")
    if relative.startswith("/") or ".." in Path(relative).parts or "\\" in relative:
        raise ValueError(f"unsafe package path: {relative}")


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        if "__pycache__" in child.parts or child.suffix in {".pyc", ".pyo"}:
            continue
        yield child


def _deployment_files() -> list[Path]:
    roots = [
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "configs",
        ROOT / "results/research_public_gsm8k_v1/data_manifest_full_v2_fuzzy",
        ROOT / ".aris/compute/budget_equivalent_v3_selections",
        ROOT / ".aris/compute/budget_equivalent_ood_v1",
        ROOT / "artifacts/phase2_v7_canary",
        ROOT / "artifacts/phase2_v7_preflight",
        ROOT
        / "tests/fixtures/phase2_v7_anchor/training_complete/adapter",
    ]
    files = [ROOT / "pyproject.toml", ROOT / "requirements-budget-equivalent.txt", ROOT / "requirements-cloud-b500.txt"]
    for root in roots:
        files.extend(_iter_files(root))
    tests = [
        *sorted((ROOT / "tests").glob("test_phase2*.py")),
        ROOT / "tests/test_identifiable_batch_backend.py",
    ]
    files.extend(path for path in tests if path.is_file())
    return sorted(set(path.resolve() for path in files))


def _scan(relative: str, payload: bytes) -> None:
    _safe_ascii(relative)
    if Path(relative).suffix.lower() in {".safetensors", ".pt", ".bin"}:
        return
    for label, pattern in (
        ("private_windows_path", WINDOWS_PRIVATE_PATH),
        ("ssh_endpoint", SSH_ENDPOINT),
        ("secret_like_value", TOKEN_VALUE),
    ):
        if pattern.search(payload):
            raise ValueError(f"{label} found in deployment file: {relative}")


def _manifest(files: list[tuple[str, bytes]], kind: str) -> dict:
    rows = [
        {"path": relative, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)}
        for relative, payload in files
    ]
    return {
        "schema_version": "phase2-v7-release-manifest-v1",
        "kind": kind,
        "file_count": len(rows),
        "files": rows,
        "ascii_paths_only": True,
        "private_windows_path_hits": 0,
        "ssh_endpoint_hits": 0,
        "secret_like_value_hits": 0,
        "gpu_accessed": False,
    }


def _write_tar(path: Path, files: list[tuple[str, bytes]]) -> dict:
    manifest = _manifest(files, "autodl_deployment")
    with tarfile.open(path, "x:gz", compresslevel=6) as archive:
        for relative, payload in files:
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        info = tarfile.TarInfo("DEPLOYMENT_MANIFEST.json")
        info.size = len(raw)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(raw))
    return manifest


def _write_review_zip(
    path: Path, deployment_manifest: dict, v7_source: Path | None
) -> dict:
    review_paths = [
        ROOT / "docs/20260828_phase2_v7_long_experiment_readiness.md",
        ROOT / "docs/20260828_phase2_v7_resume_and_failure_rules.md",
        ROOT / "docs/20260828_phase2_v7_gptpro_audit_prompt.md",
        ROOT / "configs/phase2_crossed_48cell_v7.json",
        ROOT / "configs/phase2_v7_environment_contract.json",
        ROOT / "configs/phase2_v7_legacy_batch1_contract.json",
        ROOT / "configs/phase2_v7_stop_go_rules.json",
        ROOT / "artifacts/phase2_v7_preflight/preflight_report.json",
        ROOT / "artifacts/phase2_v7_preflight/commands.jsonl",
        ROOT / "artifacts/phase2_v7_preflight/control_registry.json",
        ROOT / "artifacts/phase2_v7_preflight/semantic_code_manifest_v2.json",
        ROOT / "artifacts/phase2_v7_preflight/cpu_validation_20260828_v2.json",
    ]
    files = []
    for source in review_paths:
        relative = f"review/{source.name}"
        payload = source.read_bytes()
        _scan(relative, payload)
        files.append((relative, payload))
    if v7_source is not None:
        for source in _iter_files(v7_source.resolve()):
            relative = f"source_v7/{source.relative_to(v7_source.resolve()).as_posix()}"
            payload = source.read_bytes()
            _scan(relative, payload)
            files.append((relative, payload))
    files.append(
        (
            "review/deployment_manifest.json",
            (json.dumps(deployment_manifest, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    )
    manifest = _manifest(files, "gptpro_review")
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, payload in files:
            archive.writestr(relative, payload)
        archive.writestr(
            "REVIEW_MANIFEST.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v7-source", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment = output_dir / "phase2_v7_autodl_deployment.tar.gz"
    review = output_dir / "phase2_v7_gptpro_review.zip"
    source_files = []
    for source in _deployment_files():
        relative = source.relative_to(ROOT).as_posix()
        payload = source.read_bytes()
        _scan(relative, payload)
        source_files.append((relative, payload))
    deployment_manifest = _write_tar(deployment, source_files)
    review_manifest = _write_review_zip(review, deployment_manifest, args.v7_source)
    summary = {
        "status": "PASS",
        "deployment": {
            "path": deployment.name,
            "sha256": _sha256_bytes(deployment.read_bytes()),
            "size_bytes": deployment.stat().st_size,
            "file_count": deployment_manifest["file_count"],
        },
        "review": {
            "path": review.name,
            "sha256": _sha256_bytes(review.read_bytes()),
            "size_bytes": review.stat().st_size,
            "file_count": review_manifest["file_count"],
        },
        "gpu_accessed": False,
    }
    summary_path = output_dir / "release_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
