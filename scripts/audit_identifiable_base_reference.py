"""Independently audit the resumable four-task base-model reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract  # noqa: E402
from eg_sft.experiment.budget_equivalent_ood_audit_v2 import (  # noqa: E402
    audit_complete_ood_dataset_from_raw,
)
from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_ood_runtime import (  # noqa: E402
    OOD_DATASETS,
    resolve_ood_contract,
)
from eg_sft.experiment.identifiable_base_reference import audit_gsm_rows  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/identifiable_budget_v4_matrix.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    complete = json.loads((run_dir / "base_reference_complete.json").read_text(encoding="utf-8"))
    if complete.get("status") != "PASS" or complete.get("accuracy_withheld") is not True:
        raise ValueError("base-reference completion artifact is invalid")
    anchor = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=config_path,
        cell_id="rep1_random_common_mix_train29",
    )
    gsm_records = sorted(
        [row for row in read_jsonl(anchor["data_dir"] / "gsm8k_records.jsonl") if row["protocol_split"] == anchor["config"]["evaluation"]["split"]],
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    dataset_reports = [
        {
            "dataset": "gsm8k",
            **audit_gsm_rows(
                rows=read_jsonl(run_dir / "gsm8k" / "raw_outputs.jsonl"),
                frozen_records=gsm_records,
            ),
            "raw_outputs_sha256": file_sha256(run_dir / "gsm8k" / "raw_outputs.jsonl"),
        }
    ]
    for dataset in OOD_DATASETS:
        contract = resolve_ood_contract(
            repo_root=ROOT,
            matrix_config_path=config_path,
            dataset=dataset,
        )
        raw_path = run_dir / dataset / "raw_outputs.jsonl"
        report = audit_complete_ood_dataset_from_raw(
            rows=read_jsonl(raw_path),
            frozen_records=contract["records"],
        )
        dataset_reports.append(
            {
                "dataset": dataset,
                "status": "PASS",
                "record_count": int(report["record_count"]),
                "unique_record_id_count": int(report["unique_record_id_count"]),
                "ordered_frozen_membership": True,
                "parser_rows_recomputed_from_raw_text": int(report["parser_rows_recomputed_from_raw_text"]),
                "parser_mismatch_count": 0,
                "raw_outputs_sha256": file_sha256(raw_path),
                "accuracy_withheld": True,
            }
        )
    artifact = {
        "audit_schema_version": "identifiable-base-reference-audit-v1",
        "status": "PASS",
        "record_count": sum(int(row["record_count"]) for row in dataset_reports),
        "dataset_count": 4,
        "datasets": dataset_reports,
        "completion_sha256": file_sha256(run_dir / "base_reference_complete.json"),
        "accuracy_withheld": True,
        "gpu_accessed": False,
    }
    if artifact["record_count"] != 3841:
        raise ValueError("base-reference audit total changed")
    output = run_dir / "base_reference_audit.json"
    write_bytes_exclusive_or_verify(output, canonical_json_bytes(artifact))
    write_bytes_exclusive_or_verify(
        output.with_suffix(output.suffix + ".sha256"),
        f"{file_sha256(output)}  {output.name}\n".encode("ascii"),
    )
    print(json.dumps({"status": "PASS", "record_count": 3841, "audit_sha256": file_sha256(output), "accuracy_withheld": True}, sort_keys=True))


if __name__ == "__main__":
    main()
