"""Unified, non-Phase-1 entry for cloud qualification and its final audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import _require_clean_git_worktree  # noqa: E402

from eg_sft.experiment.budget_equivalent_qualification import (  # noqa: E402
    qualification_preflight_summary,
    resolve_qualification_contract,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


def _write_or_validate_binding(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("qualification binding changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qualification-config",
        type=Path,
        default=Path("configs/budget_equivalent_qualification_v2.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".aris/compute/budget_equivalent_qualification_v2"),
    )
    parser.add_argument("--overfit-run-dir", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.qualification_config.resolve()
    contract = resolve_qualification_contract(
        repo_root=ROOT,
        qualification_config_path=config_path,
    )
    summary = qualification_preflight_summary(contract)
    if args.contract_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    output_root = args.output_root.resolve()
    common = ["--qualification-config", str(config_path)]
    planned = {
        **summary,
        "status": "PLANNED" if args.dry_run else "STARTING",
        "commands": [
            "16-example response-only LoRA overfit and adapter reload",
            "real adapter/optimizer/scheduler/RNG checkpoint restore probe",
            "128-row resumable GSM8K canary",
            "CPU-only qualification audit",
        ],
        "automatic_phase1_start": False,
    }
    if args.dry_run:
        print(json.dumps(planned, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("single-GPU qualification requires a BF16 CUDA GPU")
    _require_clean_git_worktree()

    run_dir = args.overfit_run_dir.resolve() if args.overfit_run_dir else None
    if run_dir is None:
        before = set(output_root.glob("*/manifest.json")) if output_root.is_dir() else set()
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_gsm8k_lora_overfit.py"),
                "--config",
                str(contract["protocol_path"]),
                "--data-manifest-dir",
                str(contract["data_dir"]),
                "--output-root",
                str(output_root),
                "--examples",
                "16",
                "--epochs",
                "12",
                "--max-length",
                "512",
                "--learning-rate",
                "0.0002",
                "--gradient-accumulation",
                "4",
                "--seed",
                str(contract["protocol"]["seed"]),
            ]
        )
        after = set(output_root.glob("*/manifest.json"))
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError("qualification could not identify exactly one new overfit run")
        run_dir = created[0].parent
    binding = {
        "schema_version": "budget-equivalent-qualification-binding-v2",
        "qualification_config_sha256": contract["qualification_config_sha256"],
        "matrix_config_sha256": contract["matrix_sha256"],
        "overfit_run_manifest_sha256": file_sha256(run_dir / "manifest.json"),
        "formal_phase1_selection_consumed": False,
        "automatic_phase1_start": False,
    }
    _write_or_validate_binding(run_dir / "qualification" / "binding.json", binding)

    audit_path = run_dir / "qualification" / "qualification_audit.json"
    if not audit_path.exists():
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "probe_budget_equivalent_qualification_resume.py"),
                *common,
                "--overfit-run-dir",
                str(run_dir),
            ]
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_budget_equivalent_qualification_canary.py"),
                *common,
                "--overfit-run-dir",
                str(run_dir),
            ]
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "audit_budget_equivalent_qualification.py"),
                *common,
                "--overfit-run-dir",
                str(run_dir),
            ]
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("cloud qualification audit did not pass")
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "budget_equivalent_cloud_qualification",
                "run_dir": str(run_dir),
                "qualification_audit_sha256": file_sha256(audit_path),
                "formal_phase1_training_started": False,
                "automatic_phase1_start": False,
                "accuracy_withheld": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
