"""Compare two seed17 training anchors before authorizing the v8 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    read_jsonl,
    write_exclusive_or_verify,
)
from eg_sft.evaluation.phase2_v8_canary import (  # noqa: E402
    FULL_LEVELS,
    compare_v8_signatures,
)


def _adapter_vector(path: Path) -> tuple[np.ndarray, list[str]]:
    values = []
    keys = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            keys.append(key)
            values.append(
                handle.get_tensor(key).to(dtype=torch.float64).numpy().reshape(-1)
            )
    if not values:
        raise ValueError("training anchor adapter has no tensors")
    return np.concatenate(values), keys


def _numeric_comparison(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        raise ValueError("training anchor adapter shapes differ")
    difference = left - right
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return {
        "element_count": int(left.size),
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "flat_cosine": cosine,
        "norm_relative_difference": abs(left_norm - right_norm) / left_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-gpu0", type=Path, required=True)
    parser.add_argument("--run-dir-gpu1", type=Path, required=True)
    parser.add_argument("--anchor-audit-gpu0", type=Path, required=True)
    parser.add_argument("--anchor-audit-gpu1", type=Path, required=True)
    parser.add_argument("--anchor-signatures-gpu0", type=Path, required=True)
    parser.add_argument("--anchor-signatures-gpu1", type=Path, required=True)
    parser.add_argument("--historical-adapter", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/phase2_v8_training_anchor_protocol.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_json(args.protocol.resolve())
    run_dirs = [args.run_dir_gpu0.resolve(), args.run_dir_gpu1.resolve()]
    completions = [read_json(path / "training_anchor_complete.json") for path in run_dirs]
    if any(row.get("status") != "PASS" for row in completions):
        raise ValueError("training anchor run is incomplete")
    if completions[0]["environment_contract_sha256"] != completions[1][
        "environment_contract_sha256"
    ]:
        raise ValueError("training anchors use different environments")
    input_contracts = [read_json(path / "training_input_contract.json") for path in run_dirs]
    exact_input_fields = list(protocol["required_exact_evidence"])
    input_mismatch = [
        field
        for field in exact_input_fields
        if input_contracts[0].get(field) != input_contracts[1].get(field)
    ]
    if input_mismatch:
        raise ValueError(f"training anchor input evidence differs: {input_mismatch}")
    step_rows = [read_jsonl(path / "optimizer_step_tokens.jsonl") for path in run_dirs]
    if len(step_rows[0]) != 64 or len(step_rows[1]) != 64:
        raise ValueError("training anchor step log count changed")
    token_counts_equal = [row["response_supervision_tokens"] for row in step_rows[0]] == [
        row["response_supervision_tokens"] for row in step_rows[1]
    ]
    loss_difference = max(
        abs(
            float(left["cumulative_mean_response_token_loss"])
            - float(right["cumulative_mean_response_token_loss"])
        )
        for left, right in zip(step_rows[0], step_rows[1], strict=True)
    )
    adapter_paths = [
        path / "training_complete" / "adapter" / "adapter_model.safetensors"
        for path in run_dirs
    ]
    vectors = [_adapter_vector(path) for path in adapter_paths]
    if vectors[0][1] != vectors[1][1]:
        raise ValueError("training anchor adapter tensor keys differ")
    new_comparison = _numeric_comparison(vectors[0][0], vectors[1][0])
    historical_vector, historical_keys = _adapter_vector(args.historical_adapter.resolve())
    historical_comparisons = []
    for vector, keys in vectors:
        if keys != historical_keys:
            raise ValueError("historical and v8 adapter tensor keys differ")
        historical_comparisons.append(_numeric_comparison(historical_vector, vector))
    anchor_signature_comparison = compare_v8_signatures(
        reference=read_jsonl(args.anchor_signatures_gpu0.resolve()),
        candidate=read_jsonl(args.anchor_signatures_gpu1.resolve()),
        levels=FULL_LEVELS,
        expected_count=128,
    )
    anchor_audits = [read_json(args.anchor_audit_gpu0.resolve()), read_json(args.anchor_audit_gpu1.resolve())]
    semantic_pass = all(
        audit.get("status") == "PASS"
        and audit.get("role") == "training_anchor128"
        and audit.get("historical_bridge", {}).get("status") == "PASS"
        for audit in anchor_audits
    )
    gates = protocol["new_worker_numeric_gates"]
    checks = {
        "input_contract_exact": not input_mismatch,
        "step_token_counts_exact": token_counts_equal,
        "loss_trajectory_within_tolerance": loss_difference
        <= float(gates["loss_trajectory_max_absolute_difference"]),
        "adapter_max_abs_within_tolerance": new_comparison["max_absolute_difference"]
        <= float(gates["adapter_tensor_max_absolute_difference"]),
        "adapter_cosine_within_tolerance": new_comparison["flat_cosine"]
        >= float(gates["adapter_flat_cosine_minimum"]),
        "adapter_norm_within_tolerance": new_comparison["norm_relative_difference"]
        <= float(gates["adapter_norm_relative_difference_maximum"]),
        "semantic_128_historical_bridge": semantic_pass,
        "semantic_128_new_block_exact": anchor_signature_comparison["status"] == "PASS",
    }
    report = {
        "schema_version": "phase2-v8-training-anchor-final-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "environment_contract_sha256": completions[0]["environment_contract_sha256"],
        "loss_trajectory_max_absolute_difference": loss_difference,
        "new_worker_adapter_comparison": new_comparison,
        "historical_adapter_comparisons": historical_comparisons,
        "historical_adapter_is_external_only": True,
        "historical_token_exact_claimed": False,
        "anchor128_cross_worker_comparison": anchor_signature_comparison,
        "formal_matrix_authorized": all(checks.values()),
        "gpu_accessed_by_finalizer": False,
        "accuracy_withheld": True,
        "artifact_hashes": {
            "gpu0_anchor_complete": file_sha256(run_dirs[0] / "training_anchor_complete.json"),
            "gpu1_anchor_complete": file_sha256(run_dirs[1] / "training_anchor_complete.json"),
            "gpu0_adapter": file_sha256(adapter_paths[0]),
            "gpu1_adapter": file_sha256(adapter_paths[1]),
            "historical_adapter": file_sha256(args.historical_adapter.resolve()),
        },
    }
    output = args.output.resolve()
    write_exclusive_or_verify(output, canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError("v8 training anchor qualification failed")


if __name__ == "__main__":
    main()
