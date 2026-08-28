from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eg_sft.evaluation.phase2_v8_canary import (
    FULL_LEVELS,
    canonical_first_eos,
    compare_v8_signatures,
    validate_v8_backend_report,
)
from eg_sft.experiment.phase2_v7_environment import canonical_json_sha256
from eg_sft.experiment.phase2_v8_environment import validate_v8_environment_manifest


def _row(index: int) -> dict:
    return {
        "record_id": f"r{index}",
        "source_index": index,
        "question_sha256": f"{index:064x}",
        "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
        "prompt_token_ids": [1, index],
        "attention_mask": [1, 1],
        "raw_continuation_ids": [index, 151643, 0],
        "first_eos_continuation_ids": [index, 151643],
        "decoded_canonical_text": str(index),
        "parser_input_text": str(index),
        "parsed_number": str(index),
        "correctness": True,
        "strict_status": "ok",
        "parse_mode": "strict_final_marker",
        "parse_status": "ok",
        "gold_value": str(index),
    }


def test_first_eos_includes_eos_and_ignores_tail_pad() -> None:
    assert canonical_first_eos([1, 151643, 151643], [151643]) == [1, 151643]
    assert canonical_first_eos([1, 2], [151643]) == [1, 2]


def test_v8_signature_comparison_supports_16_and_128() -> None:
    for count in (16, 128):
        rows = [_row(index) for index in range(count)]
        assert compare_v8_signatures(
            reference=rows,
            candidate=rows,
            levels=FULL_LEVELS,
            expected_count=count,
        )["status"] == "PASS"
    rows = [_row(index) for index in range(16)]
    broken = copy.deepcopy(rows)
    broken[0]["prompt_token_ids"] = [9]
    assert compare_v8_signatures(
        reference=rows, candidate=broken, levels=FULL_LEVELS, expected_count=16
    )["status"] == "FAIL"


def test_v8_backend_report_fails_closed(tmp_path: Path) -> None:
    report = {
        "schema_version": "phase2-v8-legacy-backend-validation-v1",
        "status": "LEGACY_BATCH1_VALIDATED",
        "environment_contract_sha256": "a" * 64,
        "canary_contract_sha256": "b" * 64,
        "eval_backend": {
            "batch_size": 1,
            "padding_policy": "natural_per_example",
            "do_sample": False,
            "num_beams": 1,
            "max_input_tokens": 512,
            "max_new_tokens": 256,
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "batch_gt1_authorized": False,
        },
        "base_new_block_exact": True,
        "adapter_historical_semantic_bridge": True,
        "adapter_new_block_token_exact": True,
        "batch_gt1_authorized": False,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_v8_backend_report(report_path=path)["status"] == "LEGACY_BATCH1_VALIDATED"
    report["adapter_new_block_token_exact"] = False
    path2 = tmp_path / "failed.json"
    path2.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="token anchor"):
        validate_v8_backend_report(report_path=path2)


def test_v8_environment_rejects_wrong_gpu_sku() -> None:
    payload = {
        "schema_version": "phase2-v8-environment-v1",
        "protocol_id": "phase2-clean-common24-v8",
        "runtime_image_fingerprint_sha256": "0" * 64,
        "software": {"python_major_minor": "3.12", "torch": "2.8.0+cu128", "cuda_runtime": "12.8", "transformers": "4.57.2", "peft": "0.18.0", "accelerate": "1.12.0", "datasets": "4.4.1", "safetensors": "0.7.0", "numpy": "2.3.2", "huggingface_hub": "0.36.0", "tokenizers": "0.22.1", "pyarrow": "22.0.0", "fsspec": "2025.9.0", "dill": "0.4.0", "multiprocess": "0.70.18"},
        "gpu": {"sku": "RTX 5090", "uuid": "GPU-x", "driver_version": "1"},
        "model": {"repo_id": "Qwen/Qwen2.5-1.5B", "revision": "8faed761d45a263340a0528343f099c05c9a4323", "files_manifest_sha256": "1" * 64},
        "tokenizer": {"revision": "8faed761d45a263340a0528343f099c05c9a4323", "files_manifest_sha256": "1" * 64},
        "research": {"parent_matrix_sha256": "44d7288f4e785af61f8ebe21ec4ad1883b8b7bd542069c2fae675796724dd29a", "phase2_matrix_sha256": "2" * 64, "data_manifest_sha256": "3" * 64, "semantic_code_manifest_sha256": "4" * 64, "dataset_cache_contract_sha256": "5" * 64, "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen", "parser_policy": "strict_final_marker_then_last_numeric_fallback"},
        "numerics": {"dtype": "bfloat16", "attention_backend": "sdpa", "tf32": False, "float32_matmul_precision": "highest", "eval_batch_size": 1, "padding_policy": "natural_per_example", "deterministic_algorithms": True, "cudnn_benchmark": False, "cudnn_deterministic": True, "cublas_workspace_config": ":4096:8"},
        "training": {"micro_batch_size": 1, "gradient_accumulation": 16, "optimizer_steps": 64, "loss_normalization": "optimizer_step_response_token_sum_over_count", "pythonhashseed": "17"},
        "resources": {"power_limit_policy": "provider_default_record_only"},
    }
    payload["environment_manifest_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(ValueError, match="gpu.sku"):
        validate_v8_environment_manifest(payload)
