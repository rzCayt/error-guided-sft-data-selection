#!/usr/bin/env python3
"""Create or verify byte-identical public snapshots of current frozen configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "configs" / "frozen"
MANIFEST = SNAPSHOT_DIR / "MANIFEST.json"
SOURCE_PATHS = (
    "configs/public_gsm8k_v1.json",
    "configs/budget_equivalent_ood_v1.json",
    "configs/phase2_clean_common24_v8_canonical.json",
    "configs/phase2_v8_statistical_protocol.json",
    "configs/phase2_v8_stop_go_rules.json",
    "configs/phase2_v8_training_anchor_protocol.json",
    "configs/candidate_utility_state_dependence_protocol_frozen_20260831_v3.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_entries() -> list[dict[str, str | int]]:
    entries = []
    for source_text in SOURCE_PATHS:
        source = ROOT / source_text
        if not source.is_file():
            raise FileNotFoundError(source)
        snapshot = SNAPSHOT_DIR / source.name
        entries.append(
            {
                "source": source.relative_to(ROOT).as_posix(),
                "snapshot": snapshot.relative_to(ROOT).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
    return entries


def write_snapshots() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    entries = expected_entries()
    for entry in entries:
        source = ROOT / str(entry["source"])
        snapshot = ROOT / str(entry["snapshot"])
        shutil.copyfile(source, snapshot)
    payload = {"schema_version": "public-frozen-config-manifest-v1", "files": entries}
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WRITTEN configs={len(entries)}")


def check_snapshots() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "public-frozen-config-manifest-v1":
        raise ValueError("unexpected frozen-config manifest schema")
    expected = expected_entries()
    if payload.get("files") != expected:
        raise RuntimeError("frozen-config manifest does not match execution sources")
    for entry in expected:
        snapshot = ROOT / str(entry["snapshot"])
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        if snapshot.stat().st_size != entry["bytes"] or sha256(snapshot) != entry["sha256"]:
            raise RuntimeError(f"frozen snapshot drift: {snapshot}")
    print(f"PASS configs={len(expected)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_snapshots()
    else:
        check_snapshots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
