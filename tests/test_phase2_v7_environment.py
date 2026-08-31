from __future__ import annotations

import copy

import pytest

from eg_sft.experiment.phase2_v7_environment import (
    canonical_json_sha256,
    compare_environment_manifests,
    validate_environment_manifest,
)


def _environment(uuid: str) -> dict:
    payload = {
        "schema_version": "phase2-v7-environment-v1",
        "protocol_id": "phase2-crossed-48cell-v7",
        "runtime_image_fingerprint_sha256": "f" * 64,
        "software": {
            "python_major_minor": "3.12",
            "torch": "2.8.0+cu128",
            "cuda_runtime": "12.8",
            "transformers": "4.57.1",
            "peft": "0.17.1",
            "accelerate": "1.10.1",
            "datasets": "4.0.0",
        },
        "gpu": {
            "sku": "NVIDIA GeForce RTX 4090 D",
            "uuid": uuid,
            "driver_version": "595.71.05",
        },
        "model": {
            "repo_id": "Qwen/Qwen2.5-1.5B",
            "revision": "8faed761d45a263340a0528343f099c05c9a4323",
            "files_manifest_sha256": "1" * 64,
        },
        "tokenizer": {"revision": "8faed761", "files_manifest_sha256": "2" * 64},
        "research": {
            "parent_matrix_sha256": "44d7288f4e785af61f8ebe21ec4ad1883b8b7bd542069c2fae675796724dd29a",
            "phase2_matrix_sha256": "3" * 64,
            "data_manifest_sha256": "4" * 64,
            "semantic_code_manifest_sha256": "5" * 64,
            "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
            "parser_policy": "strict_final_marker_then_last_numeric_fallback",
        },
        "numerics": {
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "tf32": False,
            "float32_matmul_precision": "highest",
            "eval_batch_size": 1,
            "padding_policy": "natural_per_example",
        },
        "training": {
            "micro_batch_size": 1,
            "gradient_accumulation": 16,
            "optimizer_steps": 64,
            "loss_normalization": "optimizer_step_response_token_sum_over_count",
        },
    }
    payload["environment_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def test_environment_pair_allows_only_different_uuid() -> None:
    left = _environment("GPU-left")
    right = _environment("GPU-right")
    assert len(compare_environment_manifests(baseline=left, candidate=right)) == 64


def test_environment_rejects_backend_or_driver_change() -> None:
    left = _environment("GPU-left")
    for path, value in (("driver_version", "596.0"), ("sku", "RTX 5090")):
        right = _environment("GPU-right")
        right["gpu"][path] = value
        right["environment_manifest_sha256"] = canonical_json_sha256(
            {k: v for k, v in right.items() if k != "environment_manifest_sha256"}
        )
        with pytest.raises(ValueError):
            compare_environment_manifests(baseline=left, candidate=right)


def test_environment_rejects_placeholder() -> None:
    payload = _environment("GPU-left")
    payload["software"]["peft"] = "FILL_EXACT"
    payload["environment_manifest_sha256"] = canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "environment_manifest_sha256"}
    )
    with pytest.raises(ValueError, match="placeholders"):
        validate_environment_manifest(payload)


def test_environment_self_hash_is_mandatory() -> None:
    payload = _environment("GPU-left")
    broken = copy.deepcopy(payload)
    broken["training"]["gradient_accumulation"] = 8
    with pytest.raises(ValueError, match="self-hash"):
        validate_environment_manifest(broken)
