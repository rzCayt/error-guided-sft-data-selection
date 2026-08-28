"""Build a deterministic 128-item historical adapter semantic bridge."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    canonical_jsonl_bytes,
    file_sha256,
    read_jsonl,
    semantic_canary_signature,
    write_exclusive_or_verify,
)


def _read_tar_jsonl(archive_path: Path, member: str) -> list[dict]:
    with tarfile.open(archive_path, "r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"historical evidence member is missing: {member}")
        rows = []
        for line in handle.read().decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = sorted(
        _read_tar_jsonl(
            args.parent_evidence.resolve(),
            "evaluation/merged/raw_outputs.jsonl",
        ),
        key=lambda row: (int(row["source_index"]), str(row["record_id"])),
    )
    if len(source) != 1319:
        raise ValueError("historical GSM8K evidence is not complete")
    stress = read_jsonl(ROOT / "artifacts/phase2_v7_canary/archived_adapter_16.jsonl")
    stress_ids = [str(row["record_id"]) for row in stress]
    by_id = {str(row["record_id"]): row for row in source}
    remaining = [row for row in source if str(row["record_id"]) not in set(stress_ids)]
    # Deterministic quantile-like coverage over the complete held-out order.
    indices = [round(index * (len(remaining) - 1) / 111) for index in range(112)]
    stratified = [remaining[index] for index in indices]
    selected_ids = stress_ids + [str(row["record_id"]) for row in stratified]
    if len(selected_ids) != 128 or len(set(selected_ids)) != 128:
        raise ValueError("adapter semantic bridge selection is not 128 unique rows")
    signatures = []
    for record_id in selected_ids:
        signature = semantic_canary_signature(by_id[record_id])
        signature["parser_input_text"] = signature["decoded_text"]
        signatures.append(signature)
    output_root = args.output_root.resolve()
    reference = output_root / "archived_adapter_128_semantic.jsonl"
    write_exclusive_or_verify(reference, canonical_jsonl_bytes(signatures))
    manifest = {
        "schema_version": "phase2-v8-adapter128-reference-v1",
        "status": "PASS",
        "source_parent_evidence_sha256": file_sha256(args.parent_evidence.resolve()),
        "source_member": "evaluation/merged/raw_outputs.jsonl",
        "selection_rule": "existing_16_stress_then_112_deterministic_order_quantiles_v1",
        "stress_count": 16,
        "stratified_count": 112,
        "record_count": 128,
        "record_ids": selected_ids,
        "comparison_levels": [
            "decoded_text",
            "parser_input_text",
            "parsed_number",
            "correctness",
            "strict_status"
        ],
        "reference_sha256": file_sha256(reference),
        "historical_token_ids_available": False,
        "gpu_accessed": False,
    }
    manifest_path = output_root / "archived_adapter_128_manifest.json"
    write_exclusive_or_verify(manifest_path, canonical_json_bytes(manifest))
    print(
        json.dumps(
            {
                "status": "PASS",
                "record_count": 128,
                "reference_sha256": file_sha256(reference),
                "manifest_sha256": file_sha256(manifest_path),
                "gpu_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
