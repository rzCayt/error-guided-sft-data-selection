"""Collect and self-hash one Phase-2 v7 GPU environment manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
from eg_sft.experiment.phase2_v7_environment import (  # noqa: E402
    canonical_json_sha256,
    file_tree_manifest,
    validate_environment_manifest,
)


def _gpu_identity() -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("environment collector must see exactly one GPU")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 3:
        raise ValueError("unexpected nvidia-smi identity output")
    return {"sku": parts[0], "uuid": parts[1], "driver_version": parts[2]}


def _package(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(f"required package is missing: {name}") from error


def _runtime_fingerprint() -> str:
    payload = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "os_release": (
            Path("/etc/os-release").read_text(encoding="utf-8")
            if Path("/etc/os-release").is_file()
            else platform.system()
        ),
    }
    return canonical_json_sha256(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_crossed_48cell_v7.json"),
    )
    parser.add_argument("--worker-id", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--static-runtime", type=Path)
    parser.add_argument("--semantic-code-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    matrix = read_json(config_path)
    semantic = read_json(args.semantic_code_manifest.resolve())
    if semantic.get("schema_version") != "phase2-v7-semantic-code-manifest-v1":
        raise ValueError("semantic code manifest schema changed")
    if args.model_snapshot is None:
        raise ValueError("--model-snapshot is required for revision identity")
    snapshot = args.model_snapshot.resolve(strict=True)
    if not snapshot.is_dir() or snapshot.name != (
        "8faed761d45a263340a0528343f099c05c9a4323"
    ):
        raise ValueError("model snapshot must be the frozen full revision directory")
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "worker_id": args.worker_id,
                    "matrix_sha256": file_sha256(config_path),
                    "semantic_code_manifest_sha256": file_sha256(
                        args.semantic_code_manifest.resolve()
                    ),
                    "gpu_accessed": False,
                },
                sort_keys=True,
            )
        )
        return

    import torch

    output = args.output.resolve()
    model_manifest_path = output.parent / f"{args.worker_id}_model_files.json"
    if args.static_runtime is not None:
        static = read_json(args.static_runtime.resolve())
        if static.get("schema_version") != "phase2-v7-static-runtime-v1":
            raise ValueError("static runtime schema changed")
        source_manifest = Path(str(static["model_files_manifest_path"])).resolve()
        if not source_manifest.is_file() or file_sha256(source_manifest) != static[
            "model_files_manifest_sha256"
        ]:
            raise ValueError("precomputed model file manifest changed")
        write_exclusive_or_verify(model_manifest_path, source_manifest.read_bytes())
    else:
        files = file_tree_manifest(snapshot)
        write_exclusive_or_verify(model_manifest_path, canonical_json_bytes(files))
    gpu = _gpu_identity()
    required_files = matrix["parent_matrix"]
    parent = read_json((ROOT / required_files["path"]).resolve())
    data_manifest_sha = canonical_json_sha256(parent["data_manifest"])
    environment = {
        "schema_version": "phase2-v7-environment-v1",
        "protocol_id": "phase2-crossed-48cell-v7",
        "worker_id": args.worker_id,
        "runtime_image_fingerprint_sha256": _runtime_fingerprint(),
        "software": {
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "transformers": _package("transformers"),
            "peft": _package("peft"),
            "accelerate": _package("accelerate"),
            "datasets": _package("datasets"),
        },
        "gpu": gpu,
        "model": {
            "repo_id": "Qwen/Qwen2.5-1.5B",
            "revision": "8faed761d45a263340a0528343f099c05c9a4323",
            "files_manifest_sha256": file_sha256(model_manifest_path),
        },
        "tokenizer": {
            "revision": "8faed761d45a263340a0528343f099c05c9a4323",
            "files_manifest_sha256": file_sha256(model_manifest_path),
        },
        "research": {
            "parent_matrix_sha256": required_files["sha256"],
            "phase2_matrix_sha256": file_sha256(config_path),
            "data_manifest_sha256": data_manifest_sha,
            "semantic_code_manifest_sha256": file_sha256(
                args.semantic_code_manifest.resolve()
            ),
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
    environment["environment_manifest_sha256"] = canonical_json_sha256(environment)
    contract_sha = validate_environment_manifest(environment)
    write_exclusive_or_verify(output, canonical_json_bytes(environment))
    print(
        json.dumps(
            {
                "status": "PASS",
                "worker_id": args.worker_id,
                "environment_manifest_sha256": file_sha256(output),
                "environment_contract_sha256": contract_sha,
                "gpu_accessed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
