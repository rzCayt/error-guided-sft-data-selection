"""Environment and semantic-code contracts for the Phase-2 v7 extension."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "phase2-v7-environment-v1"
PROTOCOL_ID = "phase2-crossed-48cell-v7"
PARENT_GPU_SKU = "NVIDIA GeForce RTX 4090 D"

CONTRACT_PATHS = (
    "protocol_id",
    "runtime_image_fingerprint_sha256",
    "software.python_major_minor",
    "software.torch",
    "software.cuda_runtime",
    "software.transformers",
    "software.peft",
    "software.accelerate",
    "software.datasets",
    "gpu.sku",
    "gpu.driver_version",
    "model.repo_id",
    "model.revision",
    "model.files_manifest_sha256",
    "tokenizer.revision",
    "tokenizer.files_manifest_sha256",
    "research.parent_matrix_sha256",
    "research.phase2_matrix_sha256",
    "research.data_manifest_sha256",
    "research.semantic_code_manifest_sha256",
    "research.prompt_version",
    "research.parser_policy",
    "numerics.dtype",
    "numerics.attention_backend",
    "numerics.tf32",
    "numerics.float32_matmul_precision",
    "numerics.eval_batch_size",
    "numerics.padding_policy",
    "training.micro_batch_size",
    "training.gradient_accumulation",
    "training.optimizer_steps",
    "training.loss_normalization",
)


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_tree_manifest(root: Path) -> dict[str, Any]:
    """Hash an immutable model snapshot once, before a billed GPU run."""

    root = root.resolve(strict=True)
    files = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".locks/"):
            continue
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise ValueError("model snapshot manifest is empty")
    content = {"schema_version": "phase2-v7-file-tree-v1", "files": files}
    return content | {"manifest_content_sha256": canonical_json_sha256(content)}


def _lookup(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"environment field is missing: {dotted}")
        current = current[part]
    return current


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().upper()
    return (
        not normalized
        or normalized.startswith("FILL")
        or normalized.startswith("REQUIRED")
        or normalized.startswith("MATCH_")
        or normalized == "RECORD_ONLY"
    )


def environment_contract_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {path: _lookup(payload, path) for path in CONTRACT_PATHS}


def validate_environment_manifest(payload: Mapping[str, Any]) -> str:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected environment schema")
    view = environment_contract_view(payload)
    placeholders = [path for path, value in view.items() if _is_placeholder(value)]
    if placeholders:
        raise ValueError(f"environment contains placeholders: {placeholders}")
    required = {
        "protocol_id": PROTOCOL_ID,
        "software.python_major_minor": "3.12",
        "software.torch": "2.8.0+cu128",
        "software.cuda_runtime": "12.8",
        "gpu.sku": PARENT_GPU_SKU,
        "model.repo_id": "Qwen/Qwen2.5-1.5B",
        "model.revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "research.parent_matrix_sha256": (
            "44d7288f4e785af61f8ebe21ec4ad1883b8b7bd542069c2fae675796724dd29a"
        ),
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
        "training.loss_normalization": (
            "optimizer_step_response_token_sum_over_count"
        ),
    }
    for path, expected in required.items():
        if view[path] != expected:
            raise ValueError(f"environment field changed: {path}")
    gpu_uuid = str(payload.get("gpu", {}).get("uuid", ""))
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("environment GPU UUID is missing")
    manifest_sha = str(payload.get("environment_manifest_sha256", ""))
    content = dict(payload)
    content.pop("environment_manifest_sha256", None)
    if manifest_sha != canonical_json_sha256(content):
        raise ValueError("environment manifest self-hash changed")
    return canonical_json_sha256(view)


def compare_environment_manifests(
    *, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> str:
    baseline_sha = validate_environment_manifest(baseline)
    candidate_sha = validate_environment_manifest(candidate)
    if baseline_sha != candidate_sha:
        left = environment_contract_view(baseline)
        right = environment_contract_view(candidate)
        changed = [path for path in CONTRACT_PATHS if left[path] != right[path]]
        raise ValueError(f"environment contract mismatch: {changed}")
    if baseline.get("gpu", {}).get("uuid") == candidate.get("gpu", {}).get("uuid"):
        raise ValueError("dual-worker manifests unexpectedly use the same GPU UUID")
    return baseline_sha


def semantic_code_manifest(
    *, root: Path, paths: Sequence[str], parent_commit: str
) -> dict[str, Any]:
    root = root.resolve()
    files = []
    for relative in sorted(set(str(value).replace("\\", "/") for value in paths)):
        path = (root / relative).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise ValueError(f"semantic code file is missing: {relative}")
        files.append({"path": relative, "sha256": file_sha256(path)})
    if not files:
        raise ValueError("semantic code manifest is empty")
    content = {
        "schema_version": "phase2-v7-semantic-code-manifest-v1",
        "parent_run_commit": parent_commit,
        "files": files,
    }
    return content | {"manifest_content_sha256": canonical_json_sha256(content)}
