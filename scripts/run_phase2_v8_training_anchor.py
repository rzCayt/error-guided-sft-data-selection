"""Run only the frozen seed17 training path for one v8 worker anchor."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    write_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract  # noqa: E402
from eg_sft.experiment.phase2_v8_canonical_runtime import (  # noqa: E402
    require_canonical_role,
    validate_canonical_runtime,
)
from eg_sft.experiment.phase2_v8_environment import (  # noqa: E402
    validate_v8_environment_manifest,
)
from eg_sft.experiment.phase2_v8_snapshot import (  # noqa: E402
    configure_frozen_snapshot,
    current_single_gpu_identity,
    validate_snapshot_manifest,
)
from eg_sft.training.b500 import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_clean_common24_v8_canonical.json")
    )
    parser.add_argument("--worker-id", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--anchor-id", choices=("A1", "A2", "B1"), required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-files-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-files-manifest", type=Path, required=True)
    parser.add_argument("--training-input-contract-root", type=Path, required=True)
    parser.add_argument("--canonical-runtime-files", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    canonical = validate_canonical_runtime(
        repo_root=ROOT, manifest_path=args.canonical_runtime_files.resolve()
    )
    require_canonical_role(
        canonical=canonical, role="primary_matrix", actual_path=config_path
    )
    materialized_complete_path = (
        args.training_input_contract_root.resolve() / "MATERIALIZATION_COMPLETE.json"
    )
    require_canonical_role(
        canonical=canonical,
        role="materialized_contracts",
        actual_path=materialized_complete_path,
    )
    environment = read_json(args.environment_manifest.resolve())
    environment_sha = validate_v8_environment_manifest(environment)
    expected_worker = "gpu0" if args.anchor_id in {"A1", "A2"} else "gpu1"
    if args.worker_id != expected_worker or environment.get("worker_id") != args.worker_id:
        raise ValueError("v8 training anchor ID/worker mismatch")
    if environment.get("gpu", {}).get("uuid") != args.expected_gpu_uuid:
        raise ValueError("v8 training anchor expected GPU UUID changed")
    snapshot = configure_frozen_snapshot(args.model_snapshot)
    model_manifest_path = args.model_files_manifest.resolve()
    tokenizer_manifest_path = args.tokenizer_files_manifest.resolve()
    validate_snapshot_manifest(snapshot=snapshot, manifest=read_json(model_manifest_path))
    validate_snapshot_manifest(
        snapshot=snapshot, manifest=read_json(tokenizer_manifest_path)
    )
    if environment["model"]["files_manifest_sha256"] != file_sha256(model_manifest_path):
        raise ValueError("v8 anchor model tree/environment mismatch")
    if environment["tokenizer"]["files_manifest_sha256"] != file_sha256(tokenizer_manifest_path):
        raise ValueError("v8 anchor tokenizer tree/environment mismatch")
    contract = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=config_path,
        cell_id="v8_rep1_random_common_mix_train17",
    )
    expected_input = (
        args.training_input_contract_root.resolve()
        / contract["cell_id"]
        / "training_input_hashes.json"
    )
    if not expected_input.is_file():
        raise ValueError("v8 training anchor input contract is missing")
    if args.contract_only:
        print(json.dumps({"status": "READY", "anchor_id": args.anchor_id, "cell_id": contract["cell_id"], "worker_id": args.worker_id, "training_input_contract_sha256": file_sha256(expected_input), "materialized_contracts_sha256": file_sha256(materialized_complete_path), "canonical_runtime_sha256": canonical["manifest_sha256"], "environment_contract_sha256": environment_sha, "gpu_accessed": False}, sort_keys=True))
        return
    if os.environ.get("PYTHONHASHSEED") != "17":
        raise ValueError("v8 anchor requires PYTHONHASHSEED=17 before process start")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("v8 anchor requires frozen CUBLAS_WORKSPACE_CONFIG")
    import torch
    from run_b500_formal_resumable import _global_job_lock, _require_clean_git_worktree
    from run_cloud_v2_formal_cell import _resource_preflight
    from run_budget_equivalent_cell import _create_or_resume_run, _resolved_recipe, _train

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("v8 training anchor requires one BF16 CUDA GPU")
    if current_single_gpu_identity() != environment["gpu"]:
        raise ValueError("v8 training anchor current GPU/environment mismatch")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    _require_clean_git_worktree()
    os.environ["EG_SFT_TRAINING_INPUT_CONTRACT_ROOT"] = str(
        args.training_input_contract_root.resolve()
    )
    os.environ["EG_SFT_WORKER_ID"] = args.worker_id
    anchor_contract = copy.deepcopy(contract)
    anchor_contract["output_root"] = (
        args.output_root.resolve() / args.anchor_id
    )
    recipe = _resolved_recipe(anchor_contract)
    resources = _resource_preflight(anchor_contract)
    with _global_job_lock(anchor_contract["output_root"]):
        run_dir, manifest = _create_or_resume_run(
            contract=anchor_contract,
            recipe=recipe,
            resume_run_dir=args.resume_run_dir,
            command=[sys.executable, *sys.argv],
            resources=resources,
        )
        training_dir = _train(
            run_dir=run_dir,
            manifest=manifest,
            contract=anchor_contract,
            recipe=recipe,
        )
        steps = read_jsonl(run_dir / "optimizer_step_tokens.jsonl")
        report = {
            "schema_version": "phase2-v8-training-anchor-run-v1",
            "status": "PASS",
            "worker_id": args.worker_id,
            "anchor_id": args.anchor_id,
            "cell_id": contract["cell_id"],
            "run_id": manifest["run_id"],
            "environment_contract_sha256": environment_sha,
            "environment_manifest_sha256": file_sha256(
                args.environment_manifest.resolve()
            ),
            "canonical_runtime_sha256": canonical["manifest_sha256"],
            "materialized_contracts_sha256": file_sha256(
                materialized_complete_path
            ),
            "training_input_contract_sha256": file_sha256(
                run_dir / "training_input_contract.json"
            ),
            "optimizer_step_log_sha256": file_sha256(
                run_dir / "optimizer_step_tokens.jsonl"
            ),
            "optimizer_steps": len(steps),
            "instantaneous_loss_vector": [
                float(row["instantaneous_mean_response_token_loss"]) for row in steps
            ],
            "determinism": {
                "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
                "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "attention_backend": "sdpa",
            },
            "adapter_model_sha256": file_sha256(
                training_dir / "adapter" / "adapter_model.safetensors"
            ),
            "training_metrics_sha256": file_sha256(
                training_dir / "training_metrics.json"
            ),
            "historical_adapter_token_exact_claimed": False,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "gpu_accessed": True,
            "accuracy_withheld": True,
        }
        write_exclusive_or_verify(
            run_dir / "training_anchor_complete.json", canonical_json_bytes(report)
        )
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
