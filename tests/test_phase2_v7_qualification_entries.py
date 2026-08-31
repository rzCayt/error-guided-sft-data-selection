from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from eg_sft.evaluation.phase2_v7_canary import (
    CANARY_LEVELS,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    validate_legacy_backend_report,
)
from eg_sft.experiment.phase2_v7_environment import canonical_json_sha256
from eg_sft.training.b500 import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _environment(uuid: str) -> dict:
    payload = {
        "schema_version": "phase2-v7-environment-v1",
        "protocol_id": "phase2-crossed-48cell-v7",
        "worker_id": "gpu0" if uuid.endswith("0") else "gpu1",
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
        "tokenizer": {
            "revision": "8faed761d45a263340a0528343f099c05c9a4323",
            "files_manifest_sha256": "1" * 64,
        },
        "research": {
            "parent_matrix_sha256": "44d7288f4e785af61f8ebe21ec4ad1883b8b7bd542069c2fae675796724dd29a",
            "phase2_matrix_sha256": "2" * 64,
            "data_manifest_sha256": "3" * 64,
            "semantic_code_manifest_sha256": "4" * 64,
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


def _signature(index: int) -> dict:
    return {
        "record_id": f"r{index:02d}",
        "source_index": index,
        "question_sha256": f"{index:064x}",
        "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
        "raw_ids": [index, 151643],
        "first_eos_ids": [index, 151643],
        "decoded_text": f"Final answer: {index}",
        "parsed_number": str(index),
        "correctness": True,
        "strict_status": "ok",
        "parse_mode": "strict_final_marker",
        "parse_status": "ok",
        "gold_value": str(index),
    }


def _audit(role: str, environment_sha: str) -> dict:
    return {
        "schema_version": "phase2-v7-canary-audit-v1",
        "status": "PASS",
        "role": role,
        "record_count": 16,
        "exact_all_levels": True,
        "comparison_levels": list(CANARY_LEVELS),
        "environment_contract_sha256": environment_sha,
    }


def test_pair_finalizer_emits_two_bound_legacy_reports(tmp_path: Path) -> None:
    env_paths = []
    for index in range(2):
        path = tmp_path / f"env{index}.json"
        path.write_bytes(canonical_json_bytes(_environment(f"GPU-{index}")))
        env_paths.append(path)
    environment_sha = canonical_json_sha256(
        {
            path: value
            for path, value in {
                "protocol_id": "phase2-crossed-48cell-v7",
                "runtime_image_fingerprint_sha256": "f" * 64,
            }.items()
        }
    )
    # Use the validator-derived SHA rather than duplicating its full view here.
    from eg_sft.experiment.phase2_v7_environment import validate_environment_manifest

    environment_sha = validate_environment_manifest(_environment("GPU-0"))
    audit_paths = {}
    for worker in ("gpu0", "gpu1"):
        for role in ("base_model_16", "archived_adapter_16"):
            path = tmp_path / f"{worker}_{role}.json"
            path.write_bytes(canonical_json_bytes(_audit(role, environment_sha)))
            audit_paths[(worker, role)] = path
    signatures = [_signature(index) for index in range(16)]
    anchor = tmp_path / "anchor.jsonl"
    sig0 = tmp_path / "sig0.jsonl"
    sig1 = tmp_path / "sig1.jsonl"
    for path in (anchor, sig0, sig1):
        path.write_bytes(canonical_jsonl_bytes(signatures))
    output = tmp_path / "qualification"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "finalize_phase2_v7_qualification.py"),
            "--environment-gpu0",
            str(env_paths[0]),
            "--environment-gpu1",
            str(env_paths[1]),
            "--base-audit-gpu0",
            str(audit_paths[("gpu0", "base_model_16")]),
            "--base-audit-gpu1",
            str(audit_paths[("gpu1", "base_model_16")]),
            "--adapter-audit-gpu0",
            str(audit_paths[("gpu0", "archived_adapter_16")]),
            "--adapter-audit-gpu1",
            str(audit_paths[("gpu1", "archived_adapter_16")]),
            "--adapter-signatures-gpu0",
            str(sig0),
            "--adapter-signatures-gpu1",
            str(sig1),
            "--adapter-token-anchor",
            str(anchor),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for worker in ("gpu0", "gpu1"):
        report = output / f"{worker}_legacy_backend_report.json"
        assert validate_legacy_backend_report(
            report_path=report, expected_sha256=file_sha256(report)
        )["status"] == "LEGACY_BATCH1_VALIDATED"
    pair = json.loads((output / "dual_worker_qualification.json").read_text())
    assert pair["status"] == "PASS"
    assert pair["formal_matrix_authorized"] is True
