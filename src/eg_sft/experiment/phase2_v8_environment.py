"""Environment equality contract for the clean Phase-2 v8 hardware block."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eg_sft.experiment.phase2_v7_environment import (
    CONTRACT_PATHS,
    canonical_json_sha256,
    environment_contract_view,
)


SCHEMA_VERSION = "phase2-v8-environment-v1"
PROTOCOL_ID = "phase2-clean-common24-v8"
GPU_SKU = "NVIDIA GeForce RTX 4090 D"
V8_CONTRACT_PATHS = CONTRACT_PATHS + (
    "software.safetensors",
    "software.numpy",
    "software.huggingface_hub",
    "software.tokenizers",
    "software.pyarrow",
    "software.fsspec",
    "software.dill",
    "software.multiprocess",
    "numerics.deterministic_algorithms",
    "numerics.cudnn_benchmark",
    "numerics.cudnn_deterministic",
    "numerics.cublas_workspace_config",
    "training.pythonhashseed",
    "research.dataset_cache_contract_sha256",
    "resources.power_limit_policy",
)


def _v8_contract_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    base = environment_contract_view(payload)
    for dotted in V8_CONTRACT_PATHS[len(CONTRACT_PATHS) :]:
        current: Any = payload
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise ValueError(f"v8 environment field is missing: {dotted}")
            current = current[part]
        base[dotted] = current
    return base


def validate_v8_environment_manifest(payload: Mapping[str, Any]) -> str:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected v8 environment schema")
    view = _v8_contract_view(payload)
    required = {
        "protocol_id": PROTOCOL_ID,
        "software.python_major_minor": "3.12",
        "software.torch": "2.8.0+cu128",
        "software.cuda_runtime": "12.8",
        "software.transformers": "4.57.2",
        "software.peft": "0.18.0",
        "software.accelerate": "1.12.0",
        "software.datasets": "4.4.1",
        "software.safetensors": "0.7.0",
        "software.numpy": "2.3.2",
        "software.huggingface_hub": "0.36.0",
        "software.tokenizers": "0.22.1",
        "software.pyarrow": "22.0.0",
        "software.fsspec": "2025.9.0",
        "software.dill": "0.4.0",
        "software.multiprocess": "0.70.18",
        "gpu.sku": GPU_SKU,
        "model.repo_id": "Qwen/Qwen2.5-1.5B",
        "model.revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "research.parent_matrix_sha256": "44d7288f4e785af61f8ebe21ec4ad1883b8b7bd542069c2fae675796724dd29a",
        "research.prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
        "research.parser_policy": "strict_final_marker_then_last_numeric_fallback",
        "numerics.dtype": "bfloat16",
        "numerics.attention_backend": "sdpa",
        "numerics.tf32": False,
        "numerics.float32_matmul_precision": "highest",
        "numerics.eval_batch_size": 1,
        "numerics.padding_policy": "natural_per_example",
        "training.micro_batch_size": 1,
        "training.optimizer_steps": 64,
        "training.loss_normalization": "optimizer_step_response_token_sum_over_count",
        "numerics.deterministic_algorithms": True,
        "numerics.cudnn_benchmark": False,
        "numerics.cudnn_deterministic": True,
        "numerics.cublas_workspace_config": ":4096:8",
        "training.pythonhashseed": "17",
        "resources.power_limit_policy": "provider_default_record_only",
    }
    for path, expected in required.items():
        if view[path] != expected:
            raise ValueError(f"v8 environment field changed: {path}")
    dataset_cache_sha = str(view["research.dataset_cache_contract_sha256"])
    if len(dataset_cache_sha) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_cache_sha
    ):
        raise ValueError("v8 offline dataset-cache contract SHA is invalid")
    for path in V8_CONTRACT_PATHS:
        value = view[path]
        if isinstance(value, str) and (
            not value.strip()
            or value.strip().upper().startswith(("FILL", "REQUIRED", "MATCH_"))
        ):
            raise ValueError(f"v8 environment contains placeholder: {path}")
    gpu_uuid = str(payload.get("gpu", {}).get("uuid", ""))
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("v8 environment GPU UUID is missing")
    content = dict(payload)
    observed_self_hash = str(content.pop("environment_manifest_sha256", ""))
    if observed_self_hash != canonical_json_sha256(content):
        raise ValueError("v8 environment manifest self-hash changed")
    return canonical_json_sha256(view)


def compare_v8_environment_manifests(
    *, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> str:
    baseline_sha = validate_v8_environment_manifest(baseline)
    candidate_sha = validate_v8_environment_manifest(candidate)
    if baseline_sha != candidate_sha:
        left = _v8_contract_view(baseline)
        right = _v8_contract_view(candidate)
        changed = [path for path in V8_CONTRACT_PATHS if left[path] != right[path]]
        raise ValueError(f"v8 environment contract mismatch: {changed}")
    if baseline.get("gpu", {}).get("uuid") == candidate.get("gpu", {}).get("uuid"):
        raise ValueError("v8 workers unexpectedly use the same GPU UUID")
    return baseline_sha
