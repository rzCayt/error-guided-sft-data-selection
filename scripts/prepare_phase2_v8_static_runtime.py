"""Hash the frozen model snapshot before any v8 GPU qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v8_snapshot import (  # noqa: E402
    SNAPSHOT_REVISION,
    file_subset_manifest,
    tokenizer_file_names,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--semantic-code-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.model_snapshot.resolve(strict=True)
    if snapshot.name != SNAPSHOT_REVISION:
        raise ValueError("v8 static runtime model revision changed")
    semantic = read_json(args.semantic_code_manifest.resolve())
    if semantic.get("schema_version") != "phase2-v8-semantic-code-manifest-v1":
        raise ValueError("v8 semantic code manifest changed")
    output_dir = args.output_dir.resolve()
    model_manifest = output_dir / "model_files_manifest.json"
    tokenizer_manifest = output_dir / "tokenizer_files_manifest.json"
    write_exclusive_or_verify(
        model_manifest, canonical_json_bytes(file_subset_manifest(snapshot))
    )
    write_exclusive_or_verify(
        tokenizer_manifest,
        canonical_json_bytes(
            file_subset_manifest(
                snapshot, relative_names=tokenizer_file_names(snapshot)
            )
        ),
    )
    payload = {
        "schema_version": "phase2-v8-static-runtime-v1",
        "status": "PASS",
        "model_revision": snapshot.name,
        "model_files_manifest_path": str(model_manifest),
        "model_files_manifest_sha256": file_sha256(model_manifest),
        "tokenizer_files_manifest_path": str(tokenizer_manifest),
        "tokenizer_files_manifest_sha256": file_sha256(tokenizer_manifest),
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
