"""CPU-only final audit for the cloud qualification run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.b500_engineering_audit import audit_completed_evaluation  # noqa: E402
from eg_sft.experiment.budget_equivalent_qualification import (  # noqa: E402
    resolve_qualification_contract,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qualification-config",
        type=Path,
        default=Path("configs/budget_equivalent_qualification_v2.json"),
    )
    parser.add_argument("--overfit-run-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = resolve_qualification_contract(
        repo_root=ROOT,
        qualification_config_path=args.qualification_config.resolve(),
    )
    run_dir = args.overfit_run_dir.resolve()
    gates = contract["qualification"]["single_gpu_gates"]
    overfit = _read_json(run_dir / "metrics.json")
    selected = _read_json(run_dir / "selected_records.json")
    if len(selected) != len(contract["overfit_records"]):
        raise ValueError("qualification overfit record count changed")
    if float(overfit["loss_ratio_post_over_pre"]) > float(
        gates["overfit_loss_ratio_at_most"]
    ):
        raise ValueError("qualification overfit loss-ratio gate failed")
    if float(overfit["adapter_reload_loss_absolute_difference"]) > float(
        gates["adapter_reload_loss_difference_at_most"]
    ):
        raise ValueError("qualification adapter reload gate failed")

    resume_path = run_dir / "qualification" / "resume_probe" / "report.json"
    resume = _read_json(resume_path)
    for field in (
        "adapter_state_restored",
        "optimizer_state_restored",
        "scheduler_state_restored",
        "rng_state_restored",
    ):
        if resume.get(field) is not True:
            raise ValueError(f"qualification resume gate failed: {field}")

    canary_dir = run_dir / "qualification" / "canary128"
    canary_rows = read_jsonl(canary_dir / "raw_outputs.jsonl")
    canary_metrics = _read_json(canary_dir / "sealed_metrics.json")
    canary_report = audit_completed_evaluation(
        rows=canary_rows,
        frozen_records=contract["canary_records"],
        metrics=canary_metrics,
        prompt_version=contract["matrix"]["evaluation"]["prompt_version"],
    )
    canary_report.pop("recomputed_metrics", None)
    if canary_report["row_count"] != int(gates["canary_output_count"]):
        raise ValueError("qualification canary count gate failed")

    adapter_path = run_dir / "adapter" / "adapter_model.safetensors"
    report = {
        "audit_schema_version": "budget-equivalent-cloud-qualification-v2",
        "status": "PASS",
        "qualification_config_sha256": contract["qualification_config_sha256"],
        "matrix_config_sha256": contract["matrix_sha256"],
        "overfit": {
            "example_count": len(selected),
            "loss_ratio_gate_passed": True,
            "adapter_reload_gate_passed": True,
        },
        "checkpoint_resume": {
            "adapter_state_restored": True,
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "rng_state_restored": True,
            "report_sha256": file_sha256(resume_path),
        },
        "canary": canary_report,
        "ood_contracts": {
            name: {
                "record_count": len(value["records"]),
                "records_sha256": value["records_sha256"],
            }
            for name, value in contract["ood_contracts"].items()
        },
        "artifact_hashes": {
            "adapter_model": file_sha256(adapter_path),
            "overfit_metrics": file_sha256(run_dir / "metrics.json"),
            "canary_raw_outputs": file_sha256(canary_dir / "raw_outputs.jsonl"),
            "canary_metrics": file_sha256(canary_dir / "sealed_metrics.json"),
        },
        "formal_phase1_selection_consumed": False,
        "formal_phase1_training_started": False,
        "accuracy_withheld": True,
        "claim_boundary": contract["qualification"]["claim_boundary"],
    }
    output = run_dir / "qualification" / "qualification_audit.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    output.with_suffix(".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "budget_equivalent_cloud_qualification",
                "audit_sha256": file_sha256(output),
                "formal_phase1_training_started": False,
                "accuracy_withheld": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
