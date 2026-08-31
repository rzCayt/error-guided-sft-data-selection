"""CPU-only runtime hashing to finish before a GPU instance is requested."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v7_environment import (  # noqa: E402
    file_tree_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--semantic-code-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.model_snapshot.resolve(strict=True)
    if snapshot.name != "8faed761d45a263340a0528343f099c05c9a4323":
        raise ValueError("static runtime model revision changed")
    semantic = read_json(args.semantic_code_manifest.resolve())
    if semantic.get("schema_version") != "phase2-v7-semantic-code-manifest-v1":
        raise ValueError("semantic code manifest schema changed")
    output_dir = args.output_dir.resolve()
    model_manifest = output_dir / "model_files_manifest.json"
    write_exclusive_or_verify(
        model_manifest, canonical_json_bytes(file_tree_manifest(snapshot))
    )
    payload = {
        "schema_version": "phase2-v7-static-runtime-v1",
        "status": "PASS",
        "model_revision": snapshot.name,
        "model_files_manifest_path": str(model_manifest),
        "model_files_manifest_sha256": file_sha256(model_manifest),
        "semantic_code_manifest_sha256": file_sha256(
            args.semantic_code_manifest.resolve()
        ),
        "gpu_accessed": False,
    }
    output = output_dir / "static_runtime.json"
    write_exclusive_or_verify(output, canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
