"""Create immutable base and archived-adapter 16-row canary references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    CANARY_LEVELS,
    SEMANTIC_CANARY_LEVELS,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    file_sha256,
    read_json,
    read_jsonl,
    select_canary_signatures,
    select_semantic_canary_signatures,
    write_exclusive_or_verify,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--base-raw", type=Path, required=True)
    parser.add_argument("--base-metrics", type=Path, required=True)
    parser.add_argument("--adapter-raw", type=Path, required=True)
    parser.add_argument("--adapter-metrics", type=Path, required=True)
    parser.add_argument("--adapter-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eos-token-id", type=int, default=151643)
    return parser.parse_args()


def _write_reference(
    *,
    role: str,
    source_raw: Path,
    source_metrics: Path,
    signatures: list[dict],
    output_root: Path,
    adapter_model: Path | None,
    comparison_levels: tuple[str, ...],
) -> dict:
    reference_path = output_root / f"{role}.jsonl"
    write_exclusive_or_verify(reference_path, canonical_jsonl_bytes(signatures))
    metrics = read_json(source_metrics)
    if metrics.get("raw_outputs_sha256") != file_sha256(source_raw):
        raise ValueError(f"{role} source metrics do not bind raw outputs")
    manifest = {
        "schema_version": "phase2-v7-canary-reference-v1",
        "role": role,
        "record_count": 16,
        "record_ids": [row["record_id"] for row in signatures],
        "comparison_levels": list(comparison_levels),
        "reference_sha256": file_sha256(reference_path),
        "source_raw_outputs_sha256": file_sha256(source_raw),
        "source_metrics_sha256": file_sha256(source_metrics),
        "adapter_model_sha256": (
            file_sha256(adapter_model) if adapter_model is not None else None
        ),
        "fresh_process_required": True,
        "fresh_output_path_required": True,
        "batch_size": 1,
        "padding_policy": "natural_per_example",
        "batch_gt1_authorized": False,
    }
    manifest_path = output_root / f"{role}_manifest.json"
    write_exclusive_or_verify(manifest_path, canonical_json_bytes(manifest))
    return {
        "role": role,
        "reference_sha256": manifest["reference_sha256"],
        "manifest_sha256": file_sha256(manifest_path),
        "adapter_model_sha256": manifest["adapter_model_sha256"],
    }


def main() -> None:
    args = _arguments()
    selection = read_json(args.selection_manifest.resolve())
    selected_ids = selection.get("selected_record_ids")
    if not isinstance(selected_ids, list):
        raise ValueError("selection manifest lacks selected_record_ids")
    base_raw = args.base_raw.resolve()
    adapter_raw = args.adapter_raw.resolve()
    base = select_canary_signatures(
        source_rows=read_jsonl(base_raw),
        selected_record_ids=selected_ids,
        eos_token_id=args.eos_token_id,
    )
    adapter = select_semantic_canary_signatures(
        source_rows=read_jsonl(adapter_raw),
        selected_record_ids=selected_ids,
    )
    output_root = args.output_root.resolve()
    base_result = _write_reference(
        role="base_model_16",
        source_raw=base_raw,
        source_metrics=args.base_metrics.resolve(),
        signatures=base,
        output_root=output_root,
        adapter_model=None,
        comparison_levels=CANARY_LEVELS,
    )
    adapter_result = _write_reference(
        role="archived_adapter_16",
        source_raw=adapter_raw,
        source_metrics=args.adapter_metrics.resolve(),
        signatures=adapter,
        output_root=output_root,
        adapter_model=args.adapter_model.resolve(),
        comparison_levels=SEMANTIC_CANARY_LEVELS,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "selection_manifest_sha256": file_sha256(
                    args.selection_manifest.resolve()
                ),
                "base": base_result,
                "adapter": adapter_result,
                "gpu_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
