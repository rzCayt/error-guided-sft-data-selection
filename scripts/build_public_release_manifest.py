#!/usr/bin/env python3
"""Build or verify a full-file manifest for the public research release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "releases" / "v0.5-public-research" / "MANIFEST.json"
EXCLUDED_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def payload() -> dict:
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in repository_files()
    ]
    return {
        "schema_version": "public-research-release-manifest-v1",
        "release": "v0.5-public-research",
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = json.dumps(payload(), indent=2, sort_keys=True) + "\n"
    if args.write:
        if OUTPUT.exists():
            raise FileExistsError(f"refusing to overwrite release manifest: {OUTPUT}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
        print(f"WRITTEN files={len(payload()['files'])}")
        return 0
    actual = OUTPUT.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError("public release manifest is stale")
    print(f"PASS files={len(payload()['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
