"""Continuously run one disjoint half of the v4 matrix on one visible GPU.

Launch one process as ``gpu0`` and another as ``gpu1``.  Each process has a
separate lock, log directory and registry.  A cell is recorded as complete
only after both the formal and OOD audits pass.  This controller never
unblinds accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_budget_equivalent_phase1_continuous import (  # noqa: E402
    _find_run_dir,
    _package_cell,
    _read_json,
    _run_cpu_process,
    _run_gpu_with_retries,
    _run_ood_lanes,
    _sha256,
    _verified_audit,
)
from eg_sft.experiment.identifiable_budget_v4 import (  # noqa: E402
    identifiable_registry,
    validate_identifiable_matrix,
)
from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.evaluation.identifiable_batch_backend import (  # noqa: E402
    QUALIFICATION_GATES,
)


def _qualification(path: Path) -> dict:
    report = _read_json(path.resolve())
    if report.get("status") != "PASS":
        raise ValueError("batched backend qualification has not passed")
    gates = report.get("gates", {})
    if any(gates.get(name) is not True for name in QUALIFICATION_GATES):
        raise ValueError("batched backend qualification gates are incomplete")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/identifiable_budget_v4_matrix.json"),
    )
    parser.add_argument("--worker-id", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--cuda-visible-device", required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--log-root", type=Path, default=Path("/root/autodl-tmp/identifiable-v4-logs"))
    parser.add_argument("--package-root", type=Path, default=Path("/root/autodl-tmp/identifiable-v4-packages"))
    parser.add_argument("--monitor-seconds", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _read_json(config_path)
    validate_identifiable_matrix(config)
    qualification_report_path = args.qualification_report.resolve()
    _qualification(qualification_report_path)
    qualification_report_sha256 = _sha256(qualification_report_path)
    scheduled = [wave[args.worker_id] for wave in config["dual_gpu_schedule"]]
    jobs = {str(row["cell_id"]): row for row in config["job_order"]}
    if len(scheduled) != 6 or len(set(scheduled)) != 6 or any(cell not in jobs for cell in scheduled):
        raise ValueError("worker schedule must contain six unique, known cells")
    other = "gpu1" if args.worker_id == "gpu0" else "gpu0"
    if set(scheduled) & {wave[other] for wave in config["dual_gpu_schedule"]}:
        raise ValueError("the two worker schedules overlap")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_device)
    os.environ["EG_SFT_WORKER_ID"] = args.worker_id
    os.environ["EG_SFT_EVAL_BATCH_SIZE"] = str(args.evaluation_batch_size)
    os.environ["EG_SFT_QUALIFICATION_REPORT"] = str(qualification_report_path)
    os.environ["EG_SFT_QUALIFICATION_REPORT_SHA256"] = qualification_report_sha256
    run_root = (ROOT / str(config["output_root"])).resolve()
    log_root = args.log_root.resolve() / args.worker_id
    log_root.mkdir(parents=True, exist_ok=True)
    hard_stop = 80

    for cell_id in scheduled:
        log_dir = log_root / cell_id
        log_dir.mkdir(parents=True, exist_ok=True)

        def cell_complete() -> bool:
            current = _find_run_dir(run_root, cell_id)
            return current is not None and (current / "cell_complete.json").is_file()

        def cell_command() -> list[str]:
            current = _find_run_dir(run_root, cell_id)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_identifiable_budget_v4_cell.py"),
                "--config",
                str(config_path),
                "--cell-id",
                cell_id,
            ]
            if current is not None:
                command.extend(["--resume-run-dir", str(current)])
            return command

        _run_gpu_with_retries(
            command_factory=cell_command,
            complete=cell_complete,
            log_path=log_dir / "train_reload_gsm8k.log",
            event_path=log_root / "worker_events.jsonl",
            stage="train_reload_gsm8k",
            cell_id=cell_id,
            monitor_seconds=args.monitor_seconds,
            hard_stop_temperature_c=hard_stop,
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
            event_path=log_root / "worker_events.jsonl",
            monitor_seconds=args.monitor_seconds,
            hard_stop_temperature_c=hard_stop,
            max_attempts=args.max_attempts,
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
        registry = identifiable_registry(repo_root=ROOT, config_path=config_path)
        worker_artifact = {
            "registry_schema_version": "identifiable-budget-v4-worker-v1",
            "status": "AUDITED_PASS",
            "worker_id": args.worker_id,
            "cuda_visible_device": str(args.cuda_visible_device),
            "evaluation_batch_size": args.evaluation_batch_size,
            "qualification_report_sha256": qualification_report_sha256,
            "cell_id": cell_id,
            "package": package.name,
            "package_sha256": _sha256(package),
            "matrix_audited_pass_count": registry["audited_pass_count"],
            "accuracy_withheld": True,
            "automatic_unblinding": False,
        }
        output = log_dir / "AUDITED_PASS.json"
        write_bytes_exclusive_or_verify(output, canonical_json_bytes(worker_artifact))

    print(json.dumps({"status": "WORKER_COMPLETE", "worker_id": args.worker_id, "cell_count": 6, "accuracy_withheld": True}, sort_keys=True))


if __name__ == "__main__":
    main()
