"""One-command Q0 gate executed from a clean fresh release checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
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
from eg_sft.experiment.phase2_clean_common_v8 import (  # noqa: E402
    validate_clean_common_matrix,
)
from eg_sft.experiment.phase2_v8_canonical_runtime import (  # noqa: E402
    validate_canonical_runtime,
)
from eg_sft.experiment.phase2_v8_release_gate import (  # noqa: E402
    require_clean_git,
    validate_deployment_tree,
)


def _run(command: list[str], *, log_path: Path) -> dict:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    payload = (result.stdout + "\n--- STDERR ---\n" + result.stderr).encode("utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(payload)
    return {
        "command": command,
        "returncode": result.returncode,
        "log_sha256": file_sha256(log_path),
        "status": "PASS" if result.returncode == 0 else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--canonical-runtime", type=Path, required=True)
    parser.add_argument("--release-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    deployment = validate_deployment_tree(
        repo_root=ROOT, manifest_path=args.deployment_manifest.resolve()
    )
    canonical = validate_canonical_runtime(
        repo_root=ROOT, manifest_path=args.canonical_runtime.resolve()
    )
    git_commit = require_clean_git(ROOT)
    matrix_path = canonical["roles"]["primary_matrix"]["path"]
    matrix = read_json(matrix_path)
    validate_clean_common_matrix(matrix)
    materialized = read_json(
        canonical["roles"]["materialized_contracts"]["path"]
    )
    materialized_audit = read_json(
        canonical["roles"]["materialized_contract_audit"]["path"]
    )
    static_checks = {
        "deployment_complete": deployment["status"] == "PASS",
        "canonical_and_semantic_complete": canonical["semantic_validation"]["status"]
        == "PASS",
        "clean_git": len(git_commit) == 40,
        "matrix_24": len(matrix["job_order"]) == 24,
        "materialized_24": materialized.get("status") == "PASS"
        and materialized.get("cell_count") == 24,
        "materialized_audit": materialized_audit.get("status") == "PASS",
        "release_archive_present": args.release_archive.resolve().is_file(),
    }
    semantic = read_json(canonical["roles"]["semantic_code_manifest"]["path"])
    python_semantic_files = [
        row["path"] for row in semantic["files"] if str(row["path"]).endswith(".py")
    ]
    commands = []
    commands.append(
        _run(
            [sys.executable, "-m", "pytest", "-q"],
            log_path=output / "full_pytest.log",
        )
    )
    targeted = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_phase2_v8_*.py"))
    commands.append(
        _run(
            [sys.executable, "-m", "pytest", *targeted, "-q"],
            log_path=output / "targeted_v8_pytest.log",
        )
    )
    commands.append(
        _run(
            [sys.executable, "-m", "ruff", "check", *python_semantic_files],
            log_path=output / "semantic_ruff.log",
        )
    )
    commands.append(
        _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_phase2_v8_failure_injection.py",
                "tests/test_phase2_v8_worker_lease.py",
                "-q",
            ],
            log_path=output / "failure_injection.log",
        )
    )
    contract_log = output / "cell_contract_only.log"
    contract_rows = []
    for job in matrix["job_order"]:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_identifiable_budget_v4_cell.py",
                "--config",
                str(matrix_path),
                "--cell-id",
                str(job["cell_id"]),
                "--contract-only",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        contract_rows.append(
            {
                "cell_id": job["cell_id"],
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
    contract_log.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in contract_rows) + "\n",
        encoding="utf-8",
    )
    contract_pass = all(row["returncode"] == 0 for row in contract_rows)
    checks = {
        **static_checks,
        "full_pytest": commands[0]["status"] == "PASS",
        "targeted_v8_pytest": commands[1]["status"] == "PASS",
        "semantic_ruff": commands[2]["status"] == "PASS",
        "failure_injection": commands[3]["status"] == "PASS",
        "all_24_cell_contracts": contract_pass,
    }
    report = {
        "schema_version": "phase2-v8-q0-cpu-gate-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "git_commit": git_commit,
        "deployment_manifest_sha256": deployment["manifest_sha256"],
        "deployment_file_count": deployment["file_count"],
        "release_archive_sha256": file_sha256(args.release_archive.resolve()),
        "canonical_runtime_sha256": canonical["manifest_sha256"],
        "semantic_file_count": canonical["semantic_validation"]["file_count"],
        "cell_contract_log_sha256": file_sha256(contract_log),
        "commands": commands,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "gpu_accessed": False,
        "formal_matrix_authorized": False,
    }
    write_exclusive_or_verify(output / "Q0_CPU_GATE.json", canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError("Phase2 v8 Q0 CPU release gate failed")


if __name__ == "__main__":
    main()
