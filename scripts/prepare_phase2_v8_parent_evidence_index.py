"""Extract only the 16 immutable parent evidence manifests for v8 review."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    write_exclusive_or_verify,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-package-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    packages = sorted(args.parent_package_root.resolve().glob("*_evidence.tar.gz"))
    if len(packages) != 16:
        raise ValueError("v8 requires exactly 16 parent evidence packages")
    rows = []
    for package in packages:
        with tarfile.open(package, "r:gz") as archive:
            handle = archive.extractfile("EVIDENCE_MANIFEST.json")
            if handle is None:
                raise ValueError(f"parent evidence manifest missing: {package.name}")
            manifest_bytes = handle.read()
            manifest = json.loads(manifest_bytes)
        cell_id = str(manifest["cell_id"])
        output = args.output_root.resolve() / f"{cell_id}_EVIDENCE_MANIFEST.json"
        write_exclusive_or_verify(output, manifest_bytes)
        rows.append(
            {
                "cell_id": cell_id,
                "evidence_package_name": package.name,
                "evidence_package_sha256": file_sha256(package),
                "evidence_manifest_sha256": file_sha256(output),
                "file_count": int(manifest["file_count"]),
                "accuracy_withheld": bool(manifest["accuracy_withheld"]),
            }
        )
    index = {
        "schema_version": "phase2-v8-parent-evidence-index-v1",
        "status": "PASS",
        "parent_cell_count": len(rows),
        "cells": rows,
        "gpu_accessed": False,
    }
    write_exclusive_or_verify(
        args.output_root.resolve() / "PARENT_EVIDENCE_INDEX.json",
        canonical_json_bytes(index),
    )
    print(json.dumps({"status": "PASS", "parent_cell_count": 16}, sort_keys=True))


if __name__ == "__main__":
    main()
