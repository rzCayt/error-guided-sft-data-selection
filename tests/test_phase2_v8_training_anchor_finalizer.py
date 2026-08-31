from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from eg_sft.evaluation.phase2_v7_canary import canonical_jsonl_bytes


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _signature(index: int) -> dict:
    return {
        "record_id": f"r{index}",
        "source_index": index,
        "question_sha256": f"{index:064x}",
        "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
        "prompt_token_ids": [1, index],
        "attention_mask": [1, 1],
        "raw_continuation_ids": [index, 151643],
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


@pytest.mark.parametrize("historical_status", ["PASS", "FAIL"])
def test_training_anchor_finalizer_passes_identical_synthetic_workers(
    tmp_path: Path,
    historical_status: str,
) -> None:
    environment_sha = "e" * 64
    runs = []
    audit_paths = []
    signature_paths = []
    input_contract = {
        "selection_manifest_sha256": "a" * 64,
        "selected_id_order_sha256": "b" * 64,
        "ordered_sample_occurrence_sha256": "c" * 64,
        "tokenized_input_sha256": "d" * 64,
        "label_mask_sha256": "e" * 64,
        "optimizer_step_plan_sha256": "f" * 64,
        "step_response_token_counts_sha256": "1" * 64,
        "training_config_sha256": "2" * 64,
        "rng_map_sha256": "3" * 64,
    }
    for anchor_id, worker in (("A1", "gpu0"), ("A2", "gpu0"), ("B1", "gpu1")):
        run = tmp_path / anchor_id
        runs.append(run)
        _write_json(
            run / "training_anchor_complete.json",
            {"status": "PASS", "anchor_id": anchor_id, "worker_id": worker, "environment_contract_sha256": environment_sha, "canonical_runtime_sha256": "c" * 64, "materialized_contracts_sha256": "m" * 64},
        )
        _write_json(run / "training_input_contract.json", input_contract)
        steps = [
            {
                "optimizer_step": index,
                "response_supervision_tokens": 1000,
                "cumulative_mean_response_token_loss": 1.0 - index / 1000,
                "instantaneous_mean_response_token_loss": 1.0 - index / 1000,
            }
            for index in range(1, 65)
        ]
        step_path = run / "optimizer_step_tokens.jsonl"
        step_path.parent.mkdir(parents=True, exist_ok=True)
        step_path.write_text(
            "".join(json.dumps(row) + "\n" for row in steps), encoding="utf-8"
        )
        adapter = run / "training_complete/adapter/adapter_model.safetensors"
        adapter.parent.mkdir(parents=True, exist_ok=True)
        save_file({"lora": torch.tensor([1.0, 2.0])}, adapter)
        audit = tmp_path / f"{anchor_id}_audit.json"
        _write_json(
            audit,
            {
                "status": historical_status,
                "role": "training_anchor128",
                "historical_bridge": {"status": historical_status},
            },
        )
        audit_paths.append(audit)
        signatures = tmp_path / f"{anchor_id}_signatures.jsonl"
        signatures.write_bytes(
            canonical_jsonl_bytes([_signature(index) for index in range(128)])
        )
        signature_paths.append(signatures)
    historical = tmp_path / "historical.safetensors"
    save_file({"lora": torch.tensor([1.0, 2.0])}, historical)
    output = tmp_path / "final.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/finalize_phase2_v8_training_anchor_v2.py"),
            "--run-dir-a1",
            str(runs[0]),
            "--run-dir-a2",
            str(runs[1]),
            "--run-dir-b1",
            str(runs[2]),
            "--anchor-audit-a1",
            str(audit_paths[0]),
            "--anchor-audit-a2",
            str(audit_paths[1]),
            "--anchor-audit-b1",
            str(audit_paths[2]),
            "--anchor-signatures-a1",
            str(signature_paths[0]),
            "--anchor-signatures-a2",
            str(signature_paths[1]),
            "--anchor-signatures-b1",
            str(signature_paths[2]),
            "--historical-adapter",
            str(historical),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["qualification_passed"] is True
    assert report["formal_matrix_authorized"] is False
    assert report["release_go_required"] is True
    assert report["checks"]["same_gpu_signature_exact"] is True
    assert report["checks"]["cross_gpu_signature_exact"] is True
    assert report["diagnostics"]["historical_semantic_bridge_all_three"] is (
        historical_status == "PASS"
    )
    assert report["diagnostics"]["historical_failure_blocks_primary"] is False
    assert report["diagnostics"]["historical_failure_blocks_merging_parent_seed17"] is (
        historical_status == "FAIL"
    )


def test_training_anchor_ratio_is_diagnostic_not_a_hard_gate() -> None:
    source = (ROOT / "scripts/finalize_phase2_v8_training_anchor_v2.py").read_text(
        encoding="utf-8"
    )
    assert "drift_ratio_policy" in source
    assert "loss_cross_to_same_ratio" in source
    assert "adapter_max_abs_cross_to_same_ratio" in source
    assert "max(same_loss_drift" not in source
