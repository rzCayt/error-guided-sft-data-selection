"""Compare GPU0 A1/A2 and GPU1 B1 before human release review."""

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
    values, keys = [], []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            keys.append(key)
            values.append(handle.get_tensor(key).to(dtype=torch.float64).numpy().reshape(-1))
    if not values:
        raise ValueError("v8 anchor adapter is empty")
    return np.concatenate(values), keys


def _numeric(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("v8 anchor adapter is non-finite or shape changed")
    difference = left - right
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    return {
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "flat_cosine": float(np.dot(left, right) / (left_norm * right_norm)),
        "norm_relative_difference": abs(left_norm - right_norm) / left_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for anchor in ("a1", "a2", "b1"):
        parser.add_argument(f"--run-dir-{anchor}", type=Path, required=True)
        parser.add_argument(f"--anchor-audit-{anchor}", type=Path, required=True)
        parser.add_argument(f"--anchor-signatures-{anchor}", type=Path, required=True)
    parser.add_argument("--historical-adapter", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/phase2_v8_training_anchor_protocol.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    if protocol.get("schema_version") != "phase2-v8-training-anchor-v2":
        raise ValueError("unexpected v8 training-anchor protocol schema")
    protocol_sha256 = file_sha256(protocol_path)
    labels = ("A1", "A2", "B1")
    run_dirs = [getattr(args, f"run_dir_{label.lower()}").resolve() for label in labels]
    completions = [read_json(path / "training_anchor_complete.json") for path in run_dirs]
    if [row.get("anchor_id") for row in completions] != list(labels):
        raise ValueError("v8 anchor IDs/order changed")
    if any(row.get("status") != "PASS" for row in completions):
        raise ValueError("v8 anchor run is incomplete")
    if len({row["environment_contract_sha256"] for row in completions}) != 1:
        raise ValueError("v8 anchors use different environment contracts")
    if len({row.get("canonical_runtime_sha256") for row in completions}) != 1:
        raise ValueError("v8 anchors use different canonical runtime authorities")
    if len({row.get("materialized_contracts_sha256") for row in completions}) != 1:
        raise ValueError("v8 anchors use different materialized input contracts")
    if any(len(str(row.get("canonical_runtime_sha256", ""))) != 64 for row in completions):
        raise ValueError("v8 anchor canonical runtime binding is missing")
    if any(len(str(row.get("materialized_contracts_sha256", ""))) != 64 for row in completions):
        raise ValueError("v8 anchor materialized-contract binding is missing")
    exact_fields = list(protocol["required_exact_evidence"])
    inputs = [read_json(path / "training_input_contract.json") for path in run_dirs]
    input_exact = all(
        row.get(field) == inputs[0].get(field)
        for row in inputs[1:]
        for field in exact_fields
    )
    steps = [read_jsonl(path / "optimizer_step_tokens.jsonl") for path in run_dirs]
    if any(len(rows) != 64 for rows in steps):
        raise ValueError("v8 anchor must contain 64 optimizer steps")
    token_exact = all(
        [row["response_supervision_tokens"] for row in rows]
        == [row["response_supervision_tokens"] for row in steps[0]]
        for rows in steps[1:]
    )
    loss_vectors = [
        np.asarray([row["instantaneous_mean_response_token_loss"] for row in rows], dtype=float)
        for rows in steps
    ]
    if not all(np.isfinite(vector).all() for vector in loss_vectors):
        raise ValueError("v8 anchor loss vector contains NaN/Inf")
    loss_same = float(np.max(np.abs(loss_vectors[0] - loss_vectors[1])))
    loss_cross = float(np.max(np.abs(loss_vectors[0] - loss_vectors[2])))
    adapter_paths = [
        path / "training_complete/adapter/adapter_model.safetensors" for path in run_dirs
    ]
    vectors = [_adapter_vector(path) for path in adapter_paths]
    if len({tuple(keys) for _, keys in vectors}) != 1:
        raise ValueError("v8 anchor adapter tensor keys differ")
    same = _numeric(vectors[0][0], vectors[1][0])
    cross = _numeric(vectors[0][0], vectors[2][0])
    historical_vector, historical_keys = _adapter_vector(args.historical_adapter.resolve())
    if tuple(historical_keys) != tuple(vectors[0][1]):
        raise ValueError("historical adapter tensor keys differ")
    historical = [_numeric(historical_vector, vector) for vector, _ in vectors]
    signatures = [
        read_jsonl(getattr(args, f"anchor_signatures_{label.lower()}").resolve())
        for label in labels
    ]
    signature_same = compare_v8_signatures(
        reference=signatures[0], candidate=signatures[1], levels=FULL_LEVELS, expected_count=128
    )
    signature_cross = compare_v8_signatures(
        reference=signatures[0], candidate=signatures[2], levels=FULL_LEVELS, expected_count=128
    )
    audits = [read_json(getattr(args, f"anchor_audit_{label.lower()}").resolve()) for label in labels]
    semantic_pass = all(
        row.get("status") == "PASS"
        and row.get("role") == "training_anchor128"
        and row.get("historical_bridge", {}).get("status") == "PASS"
        for row in audits
    )
    gates = protocol["new_worker_numeric_gates"]
    def numeric_pass(row: dict, loss: float) -> bool:
        return (
            loss <= float(gates["loss_trajectory_max_absolute_difference"])
            and row["max_absolute_difference"] <= float(gates["adapter_tensor_max_absolute_difference"])
            and row["flat_cosine"] >= float(gates["adapter_flat_cosine_minimum"])
            and row["norm_relative_difference"] <= float(gates["adapter_norm_relative_difference_maximum"])
        )
    multiplier = float(protocol["cross_to_same_drift_multiplier_max"])
    ratio_policy = protocol.get("drift_ratio_policy", {})
    loss_floor = float(ratio_policy.get("loss_same_gpu_floor", 1e-8))
    adapter_floor = float(ratio_policy.get("adapter_same_gpu_floor", 1e-10))
    loss_ratio_applicable = loss_same > loss_floor
    adapter_ratio_applicable = same["max_absolute_difference"] > adapter_floor
    drift_context = {
        "status": str(ratio_policy.get("status", "DIAGNOSTIC_ONLY")),
        "loss_cross_to_same_ratio": (
            loss_cross / loss_same if loss_ratio_applicable else None
        ),
        "adapter_max_abs_cross_to_same_ratio": (
            cross["max_absolute_difference"] / same["max_absolute_difference"]
            if adapter_ratio_applicable
            else None
        ),
        "loss_ratio_applicable": loss_ratio_applicable,
        "adapter_ratio_applicable": adapter_ratio_applicable,
        "frozen_ratio_context_limit": multiplier,
        "loss_ratio_within_context_limit": (
            loss_cross <= multiplier * loss_same if loss_ratio_applicable else None
        ),
        "adapter_ratio_within_context_limit": (
            cross["max_absolute_difference"]
            <= multiplier * same["max_absolute_difference"]
            if adapter_ratio_applicable
            else None
        ),
        "authoritative_gates": "absolute loss/adapter thresholds and exact signatures",
    }
    checks = {
        "input_contract_exact_all_three": input_exact,
        "step_token_counts_exact_all_three": token_exact,
        "same_gpu_numeric_within_frozen_gates": numeric_pass(same, loss_same),
        "cross_gpu_numeric_within_frozen_gates": numeric_pass(cross, loss_cross),
        "same_gpu_signature_exact": signature_same["status"] == "PASS",
        "cross_gpu_signature_exact": signature_cross["status"] == "PASS",
        "historical_semantic_bridge_all_three": semantic_pass,
    }
    passed = all(checks.values())
    report = {
        "schema_version": "phase2-v8-training-anchor-final-v2",
        "training_anchor_protocol_sha256": protocol_sha256,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "environment_contract_sha256": completions[0]["environment_contract_sha256"],
        "canonical_runtime_sha256": completions[0]["canonical_runtime_sha256"],
        "materialized_contracts_sha256": completions[0]["materialized_contracts_sha256"],
        "same_gpu": {"loss_max_abs": loss_same, "adapter": same, "signature": signature_same},
        "cross_gpu": {"loss_max_abs": loss_cross, "adapter": cross, "signature": signature_cross},
        "drift_context": drift_context,
        "historical_adapter_comparisons": historical,
        "qualification_passed": passed,
        "formal_matrix_authorized": False,
        "release_go_required": True,
        "accuracy_withheld": True,
        "artifact_hashes": {
            label: {
                "completion": file_sha256(run_dirs[index] / "training_anchor_complete.json"),
                "adapter": file_sha256(adapter_paths[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    write_exclusive_or_verify(args.output.resolve(), canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("v8 A1/A2/B1 training anchor qualification failed")


if __name__ == "__main__":
    main()
