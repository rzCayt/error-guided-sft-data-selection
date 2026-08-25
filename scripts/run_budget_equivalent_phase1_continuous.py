"""Continuously execute sealed Phase 1 cells with fail-closed audit gates.

This is an operational controller only. It does not change the frozen matrix,
training recipe, prompts, parser, datasets, selectors, or unblinding policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - controller itself runs on Linux
    fcntl = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json")
DEFAULT_PACKAGE_ROOT = Path("/root/autodl-tmp/budget-v3-cell-packages")
DEFAULT_LOG_ROOT = Path("/root/autodl-tmp/budget-v3-logs/phase1-continuous")
OOD_DATASETS = ("svamp", "asdiv_numeric", "multiarith")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_hash_sidecar(artifact: Path) -> None:
    candidates = (
        artifact.with_suffix(".sha256"),
        artifact.with_suffix(artifact.suffix + ".sha256"),
    )
    sidecar = next((path for path in candidates if path.is_file()), None)
    if not artifact.is_file() or sidecar is None:
        raise ValueError(f"missing artifact or SHA sidecar: {artifact}")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    observed = _sha256(artifact)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch: {artifact}")


def _verified_audit(run_dir: Path, name: str, cell_id: str) -> bool:
    artifact = run_dir / "audit" / name
    if not artifact.is_file():
        return False
    _verify_hash_sidecar(artifact)
    report = _read_json(artifact)
    if report.get("status") != "PASS" or report.get("accuracy_withheld") is not True:
        raise ValueError(f"invalid sealed audit: {artifact}")
    reported_cell = report.get("cell_id")
    if reported_cell is not None and reported_cell != cell_id:
        raise ValueError(f"audit cell mismatch: {artifact}")
    return True


def _find_run_dir(run_root: Path, cell_id: str) -> Path | None:
    matches: list[Path] = []
    if run_root.is_dir():
        for manifest_path in sorted(run_root.glob("*/manifest.json")):
            manifest = _read_json(manifest_path)
            if manifest.get("config", {}).get("cell_id") == cell_id:
                matches.append(manifest_path.parent.resolve())
    if len(matches) > 1:
        raise ValueError(f"multiple formal run directories for {cell_id}: {matches}")
    return matches[0] if matches else None


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 5:
        raise ValueError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return {
        "temperature_c": int(float(fields[0])),
        "utilization_percent": int(float(fields[1])),
        "memory_used_mib": int(float(fields[2])),
        "memory_total_mib": int(float(fields[3])),
        "power_draw_w": float(fields[4]),
    }


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _run_gpu_process(
    *,
    command: list[str],
    log_path: Path,
    event_path: Path,
    stage: str,
    cell_id: str,
    monitor_seconds: int,
    hard_stop_temperature_c: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        log_handle.write(f"\n[{_utc_now()}] START {' '.join(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        last_monitor = time.monotonic() - monitor_seconds
        while process.poll() is None:
            time.sleep(5)
            if time.monotonic() - last_monitor < monitor_seconds:
                continue
            last_monitor = time.monotonic()
            snapshot = _gpu_snapshot()
            _append_event(
                event_path,
                {
                    "event": "gpu_monitor",
                    "recorded_at_utc": _utc_now(),
                    "cell_id": cell_id,
                    "stage": stage,
                    **snapshot,
                },
            )
            if snapshot["temperature_c"] >= hard_stop_temperature_c:
                _append_event(
                    event_path,
                    {
                        "event": "temperature_hard_stop",
                        "recorded_at_utc": _utc_now(),
                        "cell_id": cell_id,
                        "stage": stage,
                        **snapshot,
                    },
                )
                _terminate_process_group(process)
                return 86
        return int(process.returncode or 0)


def _run_cpu_process(*, command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        log_handle.write(f"\n[{_utc_now()}] START {' '.join(command)}\n")
        log_handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _run_gpu_with_retries(
    *,
    command_factory: Callable[[], list[str]],
    complete: Callable[[], bool],
    log_path: Path,
    event_path: Path,
    stage: str,
    cell_id: str,
    monitor_seconds: int,
    hard_stop_temperature_c: int,
    max_attempts: int,
) -> None:
    if complete():
        return
    for attempt in range(1, max_attempts + 1):
        _append_event(
            event_path,
            {
                "event": "stage_attempt_started",
                "recorded_at_utc": _utc_now(),
                "cell_id": cell_id,
                "stage": stage,
                "attempt": attempt,
            },
        )
        return_code = _run_gpu_process(
            command=command_factory(),
            log_path=log_path,
            event_path=event_path,
            stage=stage,
            cell_id=cell_id,
            monitor_seconds=monitor_seconds,
            hard_stop_temperature_c=hard_stop_temperature_c,
        )
        if return_code == 0 and complete():
            _append_event(
                event_path,
                {
                    "event": "stage_complete",
                    "recorded_at_utc": _utc_now(),
                    "cell_id": cell_id,
                    "stage": stage,
                    "attempt": attempt,
                },
            )
            return
        _append_event(
            event_path,
            {
                "event": "stage_attempt_failed",
                "recorded_at_utc": _utc_now(),
                "cell_id": cell_id,
                "stage": stage,
                "attempt": attempt,
                "return_code": return_code,
            },
        )
        if attempt < max_attempts:
            time.sleep(60 if return_code == 86 else 20)
    raise RuntimeError(f"{cell_id}/{stage} failed after {max_attempts} attempts")


def _worker_metrics(run_dir: Path, dataset: str) -> Path:
    return (
        run_dir
        / "evaluation"
        / "ood"
        / dataset
        / "workers"
        / "shard_00_of_01"
        / "metrics.json"
    )


def _run_ood_lanes(
    *,
    config_path: Path,
    run_dir: Path,
    cell_id: str,
    log_dir: Path,
    event_path: Path,
    monitor_seconds: int,
    hard_stop_temperature_c: int,
    max_attempts: int,
) -> None:
    def run_dataset(dataset: str) -> None:
        _run_gpu_with_retries(
            command_factory=lambda: [
                sys.executable,
                str(ROOT / "scripts" / "run_budget_equivalent_ood_eval_worker.py"),
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir),
                "--dataset",
                dataset,
                "--shard-index",
                "0",
                "--shard-count",
                "1",
            ],
            complete=lambda: _worker_metrics(run_dir, dataset).is_file(),
            log_path=log_dir / f"ood_{dataset}.log",
            event_path=event_path,
            stage=f"ood_{dataset}",
            cell_id=cell_id,
            monitor_seconds=monitor_seconds,
            hard_stop_temperature_c=hard_stop_temperature_c,
            max_attempts=max_attempts,
        )

    def short_lane() -> None:
        run_dataset("svamp")
        run_dataset("multiarith")

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="phase1-ood") as pool:
        futures = [pool.submit(run_dataset, "asdiv_numeric"), pool.submit(short_lane)]
        for future in futures:
            future.result()


def _package_cell(
    *, run_dir: Path, cell_id: str, package_root: Path, log_dir: Path
) -> Path:
    package_root.mkdir(parents=True, exist_ok=True)
    output = package_root / f"{cell_id}_evidence.tar.gz"
    if output.is_file():
        _verify_hash_sidecar(output)
        return output
    command = [
        sys.executable,
        str(ROOT / "scripts" / "package_budget_equivalent_cell_evidence.py"),
        "--run-dir",
        str(run_dir),
        "--output",
        str(output),
    ]
    for extra_log in sorted(log_dir.glob("*.log")):
        command.extend(["--extra-log", str(extra_log)])
    _run_cpu_process(command=command, log_path=log_dir / "package.log")
    _verify_hash_sidecar(output)
    return output


def _fully_audited_count(config: dict[str, Any], run_root: Path) -> int:
    count = 0
    for job in config["job_order"]:
        cell_id = str(job["cell_id"])
        run_dir = _find_run_dir(run_root, cell_id)
        if run_dir is None:
            continue
        if _verified_audit(run_dir, "formal_cell_audit.json", cell_id) and _verified_audit(
            run_dir, "ood_audit.json", cell_id
        ):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-cell")
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--monitor-seconds", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _read_json(config_path)
    if config.get("execution_policy", {}).get("accuracy_blind_until_all_audits") is not True:
        raise ValueError("continuous controller requires a sealed-accuracy matrix")
    required = int(
        config.get("execution_policy", {}).get(
            "required_audited_cells_before_unblinding", -1
        )
    )
    if required != len(config["job_order"]) or required != 16:
        raise ValueError("continuous controller is bound to the frozen 16-cell gate")
    run_root = (ROOT / str(config["output_root"])).resolve()
    event_path = args.log_root.resolve() / "controller_events.jsonl"
    args.log_root.resolve().mkdir(parents=True, exist_ok=True)
    lock_path = args.log_root.resolve() / "controller.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    if fcntl is None:
        raise RuntimeError("continuous Phase 1 controller requires Linux fcntl locking")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    jobs = list(config["job_order"])
    if args.start_cell is not None:
        indices = [i for i, job in enumerate(jobs) if job["cell_id"] == args.start_cell]
        if len(indices) != 1:
            raise ValueError(f"unknown or duplicate start cell: {args.start_cell}")
        jobs = jobs[indices[0] :]

    _append_event(
        event_path,
        {
            "event": "controller_started",
            "recorded_at_utc": _utc_now(),
            "config_sha256": _sha256(config_path),
            "required_audited_cells": required,
            "start_cell": args.start_cell,
            "accuracy_withheld": True,
        },
    )
    hard_stop = int(config["resources"]["hard_stop_temperature_c"])
    for job in jobs:
        cell_id = str(job["cell_id"])
        log_dir = args.log_root.resolve() / cell_id
        log_dir.mkdir(parents=True, exist_ok=True)
        run_dir = _find_run_dir(run_root, cell_id)
        if run_dir is not None and _verified_audit(
            run_dir, "formal_cell_audit.json", cell_id
        ) and _verified_audit(run_dir, "ood_audit.json", cell_id):
            _package_cell(
                run_dir=run_dir,
                cell_id=cell_id,
                package_root=args.package_root.resolve(),
                log_dir=log_dir,
            )
            continue

        def cell_complete() -> bool:
            current = _find_run_dir(run_root, cell_id)
            return current is not None and (current / "cell_complete.json").is_file()

        def cell_command() -> list[str]:
            current = _find_run_dir(run_root, cell_id)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_budget_equivalent_cell_v3.py"),
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
            log_path=log_dir / "train_and_gsm8k.log",
            event_path=event_path,
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
            event_path=event_path,
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
        audited = _fully_audited_count(config, run_root)
        _append_event(
            event_path,
            {
                "event": "cell_fully_audited",
                "recorded_at_utc": _utc_now(),
                "cell_id": cell_id,
                "audited_pass_count": audited,
                "required_audited_cells": required,
                "package": package.name,
                "package_sha256": _sha256(package),
                "accuracy_withheld": True,
            },
        )

    audited = _fully_audited_count(config, run_root)
    if audited != required:
        raise RuntimeError(f"controller ended with only {audited}/{required} audited cells")
    _append_event(
        event_path,
        {
            "event": "phase1_all_cells_audited",
            "recorded_at_utc": _utc_now(),
            "audited_pass_count": audited,
            "required_audited_cells": required,
            "unblinding_permitted": True,
            "accuracy_still_withheld_by_controller": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "AUDITED_PASS",
                "audited_pass_count": audited,
                "required_audited_cells": required,
                "accuracy_withheld": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
