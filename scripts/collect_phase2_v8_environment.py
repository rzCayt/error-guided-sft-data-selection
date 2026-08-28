"""Collect one clean-block v8 environment from precomputed CPU assets."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v7_environment import canonical_json_sha256  # noqa: E402
from eg_sft.experiment.phase2_v8_environment import (  # noqa: E402
    validate_v8_environment_manifest,
)


def _gpu_identity() -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("v8 environment collector must see exactly one GPU")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 3:
        raise ValueError("unexpected v8 nvidia-smi identity output")
    return {"sku": parts[0], "uuid": parts[1], "driver_version": parts[2]}


def _runtime_fingerprint() -> str:
    return canonical_json_sha256(
        {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "os_release": (
                Path("/etc/os-release").read_text(encoding="utf-8")
                if Path("/etc/os-release").is_file()
                else platform.system()
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_clean_common24_v8_canonical.json")
    )
    parser.add_argument("--worker-id", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--static-runtime", type=Path, required=True)
    parser.add_argument("--semantic-code-manifest", type=Path, required=True)
    parser.add_argument("--dataset-cache-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    snapshot = args.model_snapshot.resolve(strict=True)
    if snapshot.name != "8faed761d45a263340a0528343f099c05c9a4323":
        raise ValueError("v8 model snapshot revision changed")
    static = read_json(args.static_runtime.resolve())
    if static.get("schema_version") != "phase2-v8-static-runtime-v1":
        raise ValueError("v8 precomputed static runtime schema changed")
    source_manifest = Path(str(static["model_files_manifest_path"])).resolve()
    if not source_manifest.is_file() or file_sha256(source_manifest) != static[
        "model_files_manifest_sha256"
    ]:
        raise ValueError("v8 precomputed model manifest changed")
    source_tokenizer_manifest = Path(
        str(static["tokenizer_files_manifest_path"])
    ).resolve()
    if (
        not source_tokenizer_manifest.is_file()
        or file_sha256(source_tokenizer_manifest)
        != static["tokenizer_files_manifest_sha256"]
    ):
        raise ValueError("v8 precomputed tokenizer manifest changed")
    semantic = read_json(args.semantic_code_manifest.resolve())
    if semantic.get("schema_version") not in {
        "phase2-v7-semantic-code-manifest-v1",
        "phase2-v8-semantic-code-manifest-v1",
    }:
        raise ValueError("v8 semantic code manifest schema changed")
    dataset_cache_report = read_json(args.dataset_cache_report.resolve())
    if (
        dataset_cache_report.get("schema_version")
        != "phase2-v8-offline-dataset-cache-v1"
        or dataset_cache_report.get("status") != "PASS"
        or dataset_cache_report.get("protocol_id") != "phase2-clean-common24-v8"
        or dataset_cache_report.get("offline_mode") is not True
        or len(dataset_cache_report.get("datasets", [])) != 4
    ):
        raise ValueError("v8 offline dataset-cache qualification is incomplete")
    stable_dataset_cache = dict(dataset_cache_report)
    observed_dataset_cache_sha = str(
        stable_dataset_cache.pop("dataset_cache_contract_sha256", "")
    )
    if observed_dataset_cache_sha != canonical_json_sha256(stable_dataset_cache):
        raise ValueError("v8 offline dataset-cache report self-hash changed")
    if args.contract_only:
        print(json.dumps({"status": "READY", "worker_id": args.worker_id, "gpu_accessed": False}, sort_keys=True))
        return
    import importlib.metadata
    import torch

    if os.environ.get("PYTHONHASHSEED") != "17":
        raise ValueError("v8 environment requires PYTHONHASHSEED=17")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("v8 environment requires frozen CUBLAS_WORKSPACE_CONFIG")

    def package(name: str) -> str:
        return importlib.metadata.version(name)

    config_path = args.config.resolve()
    matrix = read_json(config_path)
    parent = read_json((ROOT / matrix["parent_matrix"]["path"]).resolve())
    output = args.output.resolve()
    copied_manifest = output.parent / f"{args.worker_id}_model_files.json"
    copied_tokenizer_manifest = (
        output.parent / f"{args.worker_id}_tokenizer_files.json"
    )
    write_exclusive_or_verify(copied_manifest, source_manifest.read_bytes())
    write_exclusive_or_verify(
        copied_tokenizer_manifest, source_tokenizer_manifest.read_bytes()
    )
    payload = {
        "schema_version": "phase2-v8-environment-v1",
        "protocol_id": "phase2-clean-common24-v8",
        "worker_id": args.worker_id,
        "runtime_image_fingerprint_sha256": _runtime_fingerprint(),
        "software": {
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "transformers": package("transformers"),
            "peft": package("peft"),
            "accelerate": package("accelerate"),
            "datasets": package("datasets"),
            "safetensors": package("safetensors"),
            "numpy": package("numpy"),
            "huggingface_hub": package("huggingface-hub"),
            "tokenizers": package("tokenizers"),
            "pyarrow": package("pyarrow"),
            "fsspec": package("fsspec"),
            "dill": package("dill"),
            "multiprocess": package("multiprocess"),
        },
        "gpu": _gpu_identity(),
        "model": {
            "repo_id": "Qwen/Qwen2.5-1.5B",
            "revision": snapshot.name,
            "files_manifest_sha256": file_sha256(copied_manifest),
        },
        "tokenizer": {
            "revision": snapshot.name,
            "files_manifest_sha256": file_sha256(copied_tokenizer_manifest),
        },
        "research": {
            "parent_matrix_sha256": matrix["parent_matrix"]["sha256"],
            "phase2_matrix_sha256": file_sha256(config_path),
            "data_manifest_sha256": canonical_json_sha256(parent["data_manifest"]),
            "semantic_code_manifest_sha256": file_sha256(args.semantic_code_manifest.resolve()),
            "dataset_cache_contract_sha256": observed_dataset_cache_sha,
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
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "training": {
            "micro_batch_size": 1,
            "gradient_accumulation": 16,
            "optimizer_steps": 64,
            "loss_normalization": "optimizer_step_response_token_sum_over_count",
            "pythonhashseed": os.environ["PYTHONHASHSEED"],
        },
        "resources": {
            "power_limit_policy": "provider_default_record_only"
        },
    }
    payload["environment_manifest_sha256"] = canonical_json_sha256(payload)
    contract_sha = validate_v8_environment_manifest(payload)
    write_exclusive_or_verify(output, canonical_json_bytes(payload))
    print(json.dumps({"status": "PASS", "environment_contract_sha256": contract_sha, "environment_manifest_sha256": file_sha256(output), "gpu_accessed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
