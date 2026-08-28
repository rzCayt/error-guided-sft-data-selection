"""Continuous disjoint worker for the clean 24-cell v8 common block."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import file_sha256, read_json  # noqa: E402
from eg_sft.evaluation.phase2_v8_canary import validate_v8_backend_report  # noqa: E402
from eg_sft.experiment.phase2_clean_common_v8 import (  # noqa: E402
    clean_common_registry,
    validate_clean_common_matrix,
)
from eg_sft.experiment.phase2_v7_control import (  # noqa: E402
    Phase2StateStore,
    validate_complete_evidence,
    worker_schedule,
)
from eg_sft.experiment.phase2_v8_canonical_runtime import (  # noqa: E402
    require_canonical_role,
    validate_canonical_runtime,
)
from eg_sft.experiment.phase2_v8_environment import (  # noqa: E402
    validate_v8_environment_manifest,
)
from eg_sft.experiment.phase2_v8_release_gate import (  # noqa: E402
    HUMAN_CONFIRMATION,
    validate_deployment_tree,
    validate_release_authorization,
)
from eg_sft.experiment.phase2_v8_snapshot import (  # noqa: E402
    configure_frozen_snapshot,
    current_single_gpu_identity,
    validate_snapshot_manifest,
)
from eg_sft.experiment.phase2_v8_worker_lease import WorkerLease  # noqa: E402


CONFIRMATION = HUMAN_CONFIRMATION


def _gpu_identity() -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("v8 worker must see exactly one GPU")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 3:
        raise ValueError("unexpected v8 GPU identity output")
    return {"sku": values[0], "uuid": values[1], "driver_version": values[2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_clean_common24_v8_canonical.json")
    )
    parser.add_argument("--canonical-runtime-files", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--release-archive", type=Path, required=True)
    parser.add_argument("--release-authorization", type=Path, required=True)
    parser.add_argument("--worker-id", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--legacy-backend-report", type=Path, required=True)
    parser.add_argument("--training-anchor-report", type=Path, required=True)
    parser.add_argument("--training-input-contract-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-files-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-files-manifest", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--monitor-seconds", type=int, default=300)
    parser.add_argument("--hard-stop-temperature-c", type=int, default=80)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume-interrupted", action="store_true")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--operator-confirmation")
    args = parser.parse_args()
    config_path = args.config.resolve()
    matrix = read_json(config_path)
    validate_clean_common_matrix(matrix)
    canonical = validate_canonical_runtime(
        repo_root=ROOT, manifest_path=args.canonical_runtime_files.resolve()
    )
    deployment = validate_deployment_tree(
        repo_root=ROOT, manifest_path=args.deployment_manifest.resolve()
    )
    require_canonical_role(
        canonical=canonical, role="primary_matrix", actual_path=config_path
    )
    scheduled = worker_schedule(matrix, args.worker_id)
    materialized_complete_path = (
        args.training_input_contract_root.resolve() / "MATERIALIZATION_COMPLETE.json"
    )
    require_canonical_role(
        canonical=canonical,
        role="materialized_contracts",
        actual_path=materialized_complete_path,
    )
    input_manifest = read_json(materialized_complete_path)
    if (
        input_manifest.get("status") != "PASS"
        or int(input_manifest.get("cell_count", -1)) != 24
        or input_manifest.get("config_sha256") != file_sha256(config_path)
    ):
        raise ValueError("v8 materialized training input contracts are incomplete")
    if args.contract_only:
        print(json.dumps({"status": "READY", "worker_id": args.worker_id, "cell_count": len(scheduled), "cells": scheduled, "canonical_runtime_sha256": canonical["manifest_sha256"], "semantic_file_count": canonical["semantic_validation"]["file_count"], "deployment_file_count": deployment["file_count"], "gpu_accessed": False, "accuracy_withheld": True}, sort_keys=True))
        return
    if args.operator_confirmation != CONFIRMATION:
        raise ValueError("exact v8 long-block operator confirmation is required")
    environment = read_json(args.environment_manifest.resolve())
    environment_sha = validate_v8_environment_manifest(environment)
    snapshot = configure_frozen_snapshot(args.model_snapshot)
    model_manifest_path = args.model_files_manifest.resolve()
    tokenizer_manifest_path = args.tokenizer_files_manifest.resolve()
    validate_snapshot_manifest(snapshot=snapshot, manifest=read_json(model_manifest_path))
    validate_snapshot_manifest(snapshot=snapshot, manifest=read_json(tokenizer_manifest_path))
    current_gpu = current_single_gpu_identity()
    for field in ("sku", "uuid", "driver_version"):
        if current_gpu[field] != environment["gpu"][field]:
            raise ValueError(f"v8 current GPU changed: {field}")
    backend_sha = file_sha256(args.legacy_backend_report.resolve())
    backend = validate_v8_backend_report(
        report_path=args.legacy_backend_report.resolve(),
        expected_sha256=backend_sha,
        expected_worker_id=args.worker_id,
        expected_gpu_uuid=current_gpu["uuid"],
    )
    anchor = read_json(args.training_anchor_report.resolve())
    if (
        anchor.get("status") != "PASS"
        or anchor.get("qualification_passed") is not True
        or anchor.get("release_go_required") is not True
        or anchor.get("environment_contract_sha256") != environment_sha
        or backend.get("environment_contract_sha256") != environment_sha
    ):
        raise ValueError("v8 training/inference qualification is incomplete")
    release_binding = validate_release_authorization(
        repo_root=ROOT,
        release_go_path=args.release_authorization.resolve(),
        worker_id=args.worker_id,
        canonical_runtime_path=args.canonical_runtime_files.resolve(),
        deployment_manifest_path=args.deployment_manifest.resolve(),
        release_archive_path=args.release_archive.resolve(),
        environment_manifest_path=args.environment_manifest.resolve(),
        backend_report_path=args.legacy_backend_report.resolve(),
        training_anchor_report_path=args.training_anchor_report.resolve(),
        model_snapshot=snapshot,
        model_manifest_path=model_manifest_path,
        tokenizer_manifest_path=tokenizer_manifest_path,
    )
    lease = WorkerLease(
        root=args.control_root.resolve() / args.worker_id,
        worker_id=args.worker_id,
        gpu_uuid=current_gpu["uuid"],
    )
    lease.acquire()
    atexit.register(lease.close)
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "EG_SFT_WORKER_ID": args.worker_id,
            "EG_SFT_EVAL_BATCH_SIZE": "1",
            "EG_SFT_LEGACY_BACKEND_REPORT": str(args.legacy_backend_report.resolve()),
            "EG_SFT_LEGACY_BACKEND_REPORT_SHA256": backend_sha,
            "EG_SFT_ENVIRONMENT_CONTRACT_SHA256": environment_sha,
            "EG_SFT_TRAINING_INPUT_CONTRACT_ROOT": str(
                args.training_input_contract_root.resolve()
            ),
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "MKL_NUM_THREADS": str(args.cpu_threads),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "17",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "EG_SFT_PHASE2_V8_RELEASE_GO": str(args.release_authorization.resolve()),
            "EG_SFT_PHASE2_V8_CANONICAL_RUNTIME": str(args.canonical_runtime_files.resolve()),
            "EG_SFT_PHASE2_V8_DEPLOYMENT_MANIFEST": str(args.deployment_manifest.resolve()),
            "EG_SFT_PHASE2_V8_RELEASE_ARCHIVE": str(args.release_archive.resolve()),
            "EG_SFT_PHASE2_V8_ENVIRONMENT_MANIFEST": str(args.environment_manifest.resolve()),
            "EG_SFT_PHASE2_V8_BACKEND_REPORT": str(args.legacy_backend_report.resolve()),
            "EG_SFT_PHASE2_V8_TRAINING_ANCHOR": str(args.training_anchor_report.resolve()),
            "EG_SFT_PHASE2_V8_MODEL_MANIFEST": str(model_manifest_path),
            "EG_SFT_PHASE2_V8_TOKENIZER_MANIFEST": str(tokenizer_manifest_path),
        }
    )
    from run_budget_equivalent_phase1_continuous import (
        _package_cell,
        _run_cpu_process,
        _run_gpu_with_retries,
        _verified_audit,
        _worker_metrics,
    )

    def unique_run_dir(cell_id: str):
        matches = []
        if run_root.is_dir():
            for manifest_path in run_root.glob("*/manifest.json"):
                manifest = read_json(manifest_path)
                if manifest.get("config", {}).get("cell_id") == cell_id:
                    matches.append(manifest_path.parent)
        if len(matches) > 1:
            raise RuntimeError(f"v8 cell has multiple run directories: {cell_id}")
        return matches[0] if matches else None

    store = Phase2StateStore(root=args.control_root, matrix_path=config_path)
    store.initialize()
    run_root = (ROOT / matrix["output_root"]).resolve()
    worker_logs = args.log_root.resolve() / args.worker_id
    worker_logs.mkdir(parents=True, exist_ok=True)
    event_path = worker_logs / "worker_events.jsonl"
    for cell_id in scheduled:
        state = store.read_state(cell_id)
        if state["state"] == "COMPLETE":
            validate_complete_evidence(
                state=state,
                required_hashes=("formal_audit_sha256", "ood_audit_sha256", "evidence_package_sha256"),
            )
            evidence = state["evidence"]
            package = args.package_root.resolve() / str(evidence["evidence_package_name"])
            if (
                not package.is_file()
                or file_sha256(package) != evidence["evidence_package_sha256"]
            ):
                raise ValueError("v8 COMPLETE evidence package changed")
            continue
        if state["state"] in {"LOCKED", "RUNNING", "AUDITING"}:
            if not args.resume_interrupted:
                raise RuntimeError(f"v8 interrupted cell requires inspection: {cell_id}")
            store.fail(
                cell_id=cell_id,
                worker_id=args.worker_id,
                attempt_id=str(state["attempt_id"]),
                reason="v8 worker restart after prefix inspection",
                evidence={"previous_state": state["state"]},
            )
        attempt_id = uuid.uuid4().hex
        store.transition(cell_id=cell_id, target="LOCKED", worker_id=args.worker_id, attempt_id=attempt_id, reason="v8 human-release-authorized worker acquired cell", evidence={"environment_contract_sha256": environment_sha, "backend_report_sha256": backend_sha, "training_anchor_report_sha256": file_sha256(args.training_anchor_report.resolve()), "canonical_runtime_sha256": canonical["manifest_sha256"], "release_go_sha256": release_binding["release_go_sha256"], "deployment_manifest_sha256": deployment["manifest_sha256"]})
        store.transition(cell_id=cell_id, target="RUNNING", worker_id=args.worker_id, attempt_id=attempt_id, reason="v8 frozen cell started")
        log_dir = worker_logs / cell_id
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            def cell_complete() -> bool:
                run = unique_run_dir(cell_id)
                return run is not None and (run / "cell_complete.json").is_file()

            def cell_command() -> list[str]:
                run = unique_run_dir(cell_id)
                command = [sys.executable, str(ROOT / "scripts/run_identifiable_budget_v4_cell.py"), "--config", str(config_path), "--cell-id", cell_id]
                if run is not None:
                    command.extend(["--resume-run-dir", str(run)])
                return command

            _run_gpu_with_retries(command_factory=cell_command, complete=cell_complete, log_path=log_dir / "train_reload_gsm8k.log", event_path=event_path, stage="train_reload_gsm8k_sequential", cell_id=cell_id, monitor_seconds=args.monitor_seconds, hard_stop_temperature_c=args.hard_stop_temperature_c, max_attempts=args.max_attempts)
            run_dir = unique_run_dir(cell_id)
            if run_dir is None:
                raise RuntimeError("v8 completed cell has no run directory")
            if not _verified_audit(run_dir, "formal_cell_audit.json", cell_id):
                _run_cpu_process(command=[sys.executable, str(ROOT / "scripts/audit_budget_equivalent_cell_v5.py"), "--config", str(config_path), "--cell-id", cell_id, "--run-dir", str(run_dir)], log_path=log_dir / "formal_audit.log")
            if not _verified_audit(run_dir, "formal_cell_audit.json", cell_id):
                raise RuntimeError("v8 formal audit did not pass")
            # Strictly one inference process per physical GPU: OOD datasets are sequential.
            for dataset in ("svamp", "asdiv_numeric", "multiarith"):
                _run_gpu_with_retries(
                    command_factory=lambda dataset=dataset: [sys.executable, str(ROOT / "scripts/run_budget_equivalent_ood_eval_worker.py"), "--config", str(config_path), "--run-dir", str(run_dir), "--dataset", dataset, "--shard-index", "0", "--shard-count", "1"],
                    complete=lambda dataset=dataset: _worker_metrics(run_dir, dataset).is_file(),
                    log_path=log_dir / f"ood_{dataset}.log",
                    event_path=event_path,
                    stage=f"ood_{dataset}_sequential",
                    cell_id=cell_id,
                    monitor_seconds=args.monitor_seconds,
                    hard_stop_temperature_c=args.hard_stop_temperature_c,
                    max_attempts=args.max_attempts,
                )
            store.transition(cell_id=cell_id, target="AUDITING", worker_id=args.worker_id, attempt_id=attempt_id, reason="v8 all GPU stages complete")
            if not _verified_audit(run_dir, "ood_audit.json", cell_id):
                _run_cpu_process(command=[sys.executable, str(ROOT / "scripts/audit_budget_equivalent_ood_v3.py"), "--config", str(config_path), "--run-dir", str(run_dir), "--shard-count", "1"], log_path=log_dir / "ood_audit.log")
            if not _verified_audit(run_dir, "ood_audit.json", cell_id):
                raise RuntimeError("v8 OOD audit did not pass")
            package = _package_cell(run_dir=run_dir, cell_id=cell_id, package_root=args.package_root.resolve(), log_dir=log_dir)
            store.transition(cell_id=cell_id, target="COMPLETE", worker_id=args.worker_id, attempt_id=attempt_id, reason="v8 formal/OOD/package passed", evidence={"run_dir_name": run_dir.name, "formal_audit_sha256": file_sha256(run_dir / "audit/formal_cell_audit.json"), "ood_audit_sha256": file_sha256(run_dir / "audit/ood_audit.json"), "evidence_package_name": package.name, "evidence_package_sha256": file_sha256(package)})
        except Exception as error:
            latest = store.read_state(cell_id)
            if latest["state"] in {"LOCKED", "RUNNING", "AUDITING"}:
                store.fail(cell_id=cell_id, worker_id=args.worker_id, attempt_id=attempt_id, reason=f"{type(error).__name__}: {error}")
            raise
    registry = clean_common_registry(repo_root=ROOT, config_path=config_path)
    print(json.dumps({"status": "WORKER_COMPLETE", "worker_id": args.worker_id, "cell_count": len(scheduled), "artifact_audited_pass_count": registry["audited_pass_count"], "accuracy_withheld": True, "automatic_unblinding": False}, sort_keys=True))


if __name__ == "__main__":
    main()
