"""Run one disjoint 16-cell Phase-2 v7 worker after explicit qualification.

This controller never changes the frozen matrix and never unblinds accuracy.
It can continue across its assigned cells only when the operator passes the
exact long-block confirmation string.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    file_sha256,
    read_json,
    validate_legacy_backend_report,
)
from eg_sft.experiment.phase2_crossed_v7 import (  # noqa: E402
    phase2_registry,
    validate_phase2_matrix,
)
from eg_sft.experiment.phase2_v7_control import (  # noqa: E402
    Phase2StateStore,
    validate_complete_evidence,
    worker_schedule,
)
from eg_sft.experiment.phase2_v7_environment import (  # noqa: E402
    validate_environment_manifest,
)


CONFIRMATION = "PHASE2_V7_32CELL_BLOCK_APPROVED"


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
        raise ValueError("worker must see exactly one GPU")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 3:
        raise ValueError("unexpected nvidia-smi identity output")
    return {"sku": parts[0], "uuid": parts[1], "driver_version": parts[2]}


def _validate_runtime_bindings(
    *, environment_manifest: Path, legacy_report: Path
) -> tuple[dict, dict, str, str]:
    environment = read_json(environment_manifest.resolve())
    environment_contract_sha = validate_environment_manifest(environment)
    report_sha = file_sha256(legacy_report.resolve())
    report = validate_legacy_backend_report(
        report_path=legacy_report.resolve(), expected_sha256=report_sha
    )
    if report["environment_contract_sha256"] != environment_contract_sha:
        raise ValueError("legacy report and environment contract differ")
    current = _gpu_identity()
    for field in ("sku", "uuid", "driver_version"):
        if current[field] != environment["gpu"][field]:
            raise ValueError(f"current GPU environment changed: {field}")
    return environment, report, environment_contract_sha, report_sha


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_crossed_48cell_v7.json"),
    )
    parser.add_argument("--worker-id", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--cuda-visible-device", default="0")
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--legacy-backend-report", type=Path, required=True)
    parser.add_argument(
        "--control-root", type=Path, default=Path("/root/autodl-tmp/phase2-v7-control")
    )
    parser.add_argument(
        "--log-root", type=Path, default=Path("/root/autodl-tmp/phase2-v7-logs")
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("/root/autodl-tmp/phase2-v7-packages"),
    )
    parser.add_argument("--monitor-seconds", type=int, default=300)
    parser.add_argument("--hard-stop-temperature-c", type=int, default=80)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume-interrupted", action="store_true")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--operator-confirmation")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config_path = args.config.resolve()
    matrix = read_json(config_path)
    validate_phase2_matrix(matrix)
    scheduled = worker_schedule(matrix, args.worker_id)
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "stage": "phase2_v7_worker_contract",
                    "worker_id": args.worker_id,
                    "cell_count": len(scheduled),
                    "cell_ids": scheduled,
                    "matrix_sha256": file_sha256(config_path),
                    "gpu_accessed": False,
                    "accuracy_withheld": True,
                },
                sort_keys=True,
            )
        )
        return
    if args.operator_confirmation != CONFIRMATION:
        raise ValueError("exact long-block operator confirmation is required")
    if not 1 <= args.cpu_threads <= 16:
        raise ValueError("cpu-threads must stay between 1 and 16")

    environment, _, environment_contract_sha, legacy_report_sha = (
        _validate_runtime_bindings(
            environment_manifest=args.environment_manifest,
            legacy_report=args.legacy_backend_report,
        )
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_device)
    os.environ["EG_SFT_WORKER_ID"] = args.worker_id
    os.environ["EG_SFT_EVAL_BATCH_SIZE"] = "1"
    os.environ["EG_SFT_LEGACY_BACKEND_REPORT"] = str(
        args.legacy_backend_report.resolve()
    )
    os.environ["EG_SFT_LEGACY_BACKEND_REPORT_SHA256"] = legacy_report_sha
    os.environ["EG_SFT_ENVIRONMENT_CONTRACT_SHA256"] = environment_contract_sha
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Import orchestration helpers only after contract validation so CPU-only
    # tests do not import torch/peft/transformers.
    from run_budget_equivalent_phase1_continuous import (
        _find_run_dir,
        _package_cell,
        _run_cpu_process,
        _run_gpu_with_retries,
        _run_ood_lanes,
        _verified_audit,
    )

    store = Phase2StateStore(root=args.control_root, matrix_path=config_path)
    store.initialize()
    run_root = (ROOT / str(matrix["output_root"])).resolve()
    worker_log_root = args.log_root.resolve() / args.worker_id
    worker_log_root.mkdir(parents=True, exist_ok=True)
    event_path = worker_log_root / "worker_events.jsonl"

    for cell_id in scheduled:
        current = store.read_state(cell_id)
        if current["state"] == "COMPLETE":
            validate_complete_evidence(
                state=current,
                required_hashes=(
                    "formal_audit_sha256",
                    "ood_audit_sha256",
                    "evidence_package_sha256",
                ),
            )
            continue
        if current["state"] in {"LOCKED", "RUNNING", "AUDITING"}:
            if not args.resume_interrupted:
                raise RuntimeError(
                    f"{cell_id} is interrupted; pass --resume-interrupted after inspection"
                )
            old_attempt = str(current["attempt_id"])
            store.fail(
                cell_id=cell_id,
                worker_id=args.worker_id,
                attempt_id=old_attempt,
                reason="worker restart after artifact-prefix inspection",
                evidence={"previous_state": current["state"]},
            )
            current = store.read_state(cell_id)
        if current["state"] not in {"PLANNED", "FAILED"}:
            raise RuntimeError(f"unexpected starting state for {cell_id}: {current['state']}")
        attempt_id = uuid.uuid4().hex
        store.transition(
            cell_id=cell_id,
            target="LOCKED",
            worker_id=args.worker_id,
            attempt_id=attempt_id,
            reason="operator-approved worker acquired cell",
            evidence={
                "environment_manifest_sha256": file_sha256(
                    args.environment_manifest.resolve()
                ),
                "environment_contract_sha256": environment_contract_sha,
                "legacy_backend_report_sha256": legacy_report_sha,
                "gpu_uuid": environment["gpu"]["uuid"],
            },
        )
        store.transition(
            cell_id=cell_id,
            target="RUNNING",
            worker_id=args.worker_id,
            attempt_id=attempt_id,
            reason="frozen train/reload/evaluation pipeline started",
        )
        log_dir = worker_log_root / cell_id
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            def cell_complete() -> bool:
                current_run = _find_run_dir(run_root, cell_id)
                return current_run is not None and (
                    current_run / "cell_complete.json"
                ).is_file()

            def cell_command() -> list[str]:
                current_run = _find_run_dir(run_root, cell_id)
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_identifiable_budget_v4_cell.py"),
                    "--config",
                    str(config_path),
                    "--cell-id",
                    cell_id,
                ]
                if current_run is not None:
                    command.extend(["--resume-run-dir", str(current_run)])
                return command

            _run_gpu_with_retries(
                command_factory=cell_command,
                complete=cell_complete,
                log_path=log_dir / "train_reload_gsm8k.log",
                event_path=event_path,
                stage="train_reload_gsm8k",
                cell_id=cell_id,
                monitor_seconds=args.monitor_seconds,
                hard_stop_temperature_c=args.hard_stop_temperature_c,
                max_attempts=args.max_attempts,
            )
            run_dir = _find_run_dir(run_root, cell_id)
            if run_dir is None:
                raise RuntimeError(f"completed cell has no run directory: {cell_id}")
            if not _verified_audit(run_dir, "formal_cell_audit.json", cell_id):
                _run_cpu_process(
                    command=[
                        sys.executable,
                        str(ROOT / "scripts" / "audit_budget_equivalent_cell_v5.py"),
                        "--config",
                        str(config_path),
                        "--cell-id",
                        cell_id,
                        "--run-dir",
                        str(run_dir),
                    ],
                    log_path=log_dir / "formal_audit.log",
                )
            if not _verified_audit(run_dir, "formal_cell_audit.json", cell_id):
                raise RuntimeError(f"formal audit did not pass: {cell_id}")
            _run_ood_lanes(
                config_path=config_path,
                run_dir=run_dir,
                cell_id=cell_id,
                log_dir=log_dir,
                event_path=event_path,
                monitor_seconds=args.monitor_seconds,
                hard_stop_temperature_c=args.hard_stop_temperature_c,
                max_attempts=args.max_attempts,
            )
            store.transition(
                cell_id=cell_id,
                target="AUDITING",
                worker_id=args.worker_id,
                attempt_id=attempt_id,
                reason="all GPU stages complete; final OOD audit and packaging started",
                evidence={"run_dir_name": run_dir.name},
            )
            if not _verified_audit(run_dir, "ood_audit.json", cell_id):
                _run_cpu_process(
                    command=[
                        sys.executable,
                        str(ROOT / "scripts" / "audit_budget_equivalent_ood_v3.py"),
                        "--config",
                        str(config_path),
                        "--run-dir",
                        str(run_dir),
                        "--shard-count",
                        "1",
                    ],
                    log_path=log_dir / "ood_audit.log",
                )
            if not _verified_audit(run_dir, "ood_audit.json", cell_id):
                raise RuntimeError(f"OOD audit did not pass: {cell_id}")
            package = _package_cell(
                run_dir=run_dir,
                cell_id=cell_id,
                package_root=args.package_root.resolve(),
                log_dir=log_dir,
            )
            formal_path = run_dir / "audit" / "formal_cell_audit.json"
            ood_path = run_dir / "audit" / "ood_audit.json"
            store.transition(
                cell_id=cell_id,
                target="COMPLETE",
                worker_id=args.worker_id,
                attempt_id=attempt_id,
                reason="formal audit, OOD audit and evidence package passed",
                evidence={
                    "run_dir_name": run_dir.name,
                    "formal_audit_sha256": file_sha256(formal_path),
                    "ood_audit_sha256": file_sha256(ood_path),
                    "evidence_package_name": package.name,
                    "evidence_package_sha256": file_sha256(package),
                    "environment_contract_sha256": environment_contract_sha,
                    "legacy_backend_report_sha256": legacy_report_sha,
                },
            )
        except Exception as error:
            latest = store.read_state(cell_id)
            if latest["state"] in {"LOCKED", "RUNNING", "AUDITING"}:
                store.fail(
                    cell_id=cell_id,
                    worker_id=args.worker_id,
                    attempt_id=attempt_id,
                    reason=f"{type(error).__name__}: {error}",
                )
            raise

    runtime_registry = phase2_registry(repo_root=ROOT, config_path=config_path)
    control_registry = store.registry()
    print(
        json.dumps(
            {
                "status": "WORKER_COMPLETE",
                "worker_id": args.worker_id,
                "cell_count": len(scheduled),
                "control_state_counts": control_registry["state_counts"],
                "artifact_audited_pass_count": runtime_registry["audited_pass_count"],
                "accuracy_withheld": True,
                "automatic_unblinding": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
