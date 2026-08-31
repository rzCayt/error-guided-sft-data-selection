"""Hash every code/config file that can change v8 scientific semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v7_environment import canonical_json_sha256  # noqa: E402


PATHS = (
    "configs/phase2_clean_common24_v8_canonical.json",
    "configs/phase2_v8_statistical_protocol.json",
    "configs/phase2_v8_training_anchor_protocol.json",
    "configs/phase2_v8_canary_contract.json",
    "configs/phase2_v8_stop_go_rules.json",
    "scripts/run_budget_equivalent_cell.py",
    "scripts/run_budget_equivalent_cell_v3.py",
    "scripts/run_budget_equivalent_eval_worker.py",
    "scripts/run_cloud_v2_formal_eval_worker.py",
    "scripts/run_budget_equivalent_ood_eval_worker.py",
    "scripts/audit_budget_equivalent_cell_v5.py",
    "scripts/audit_budget_equivalent_ood_v3.py",
    "scripts/materialize_phase2_v8_contracts.py",
    "scripts/audit_phase2_v8_materialized_contracts.py",
    "scripts/prepare_phase2_v8_static_runtime.py",
    "scripts/build_phase2_v8_canonical_runtime.py",
    "scripts/collect_phase2_v8_environment.py",
    "scripts/stage_phase2_v8_offline_datasets.py",
    "scripts/qualify_phase2_v8_offline_datasets.py",
    "scripts/run_phase2_v8_canary.py",
    "scripts/finalize_phase2_v8_qualification.py",
    "scripts/run_phase2_v8_training_anchor.py",
    "scripts/finalize_phase2_v8_training_anchor_v2.py",
    "scripts/finalize_phase2_v8_release_go.py",
    "scripts/authorize_phase2_v8_release.py",
    "scripts/phase2_v8_cpu_release_gate.py",
    "scripts/phase2_v8_cpu_release_gate.sh",
    "scripts/run_phase2_v8_worker.py",
    "scripts/aggregate_phase2_v8_progress.py",
    "scripts/analyze_phase2_v8_unblinded.py",
    "scripts/phase2_v8_prepare_host.sh",
    "scripts/phase2_v8_qualify_gpu0.sh",
    "scripts/phase2_v8_qualify_gpu1.sh",
    "scripts/phase2_v8_training_anchor_worker.sh",
    "scripts/phase2_v8_training_anchor_canary.sh",
    "scripts/phase2_v8_run_worker.sh",
    "src/eg_sft/evaluation/identifiable_batch_backend.py",
    "src/eg_sft/evaluation/phase2_v8_canary.py",
    "src/eg_sft/experiment/phase2_clean_common_v8.py",
    "src/eg_sft/experiment/phase2_v7_control.py",
    "src/eg_sft/experiment/phase2_v8_contract_audit.py",
    "src/eg_sft/experiment/phase2_v8_environment.py",
    "src/eg_sft/experiment/phase2_v8_canonical_runtime.py",
    "src/eg_sft/experiment/phase2_v8_statistics.py",
    "src/eg_sft/experiment/phase2_v8_snapshot.py",
    "src/eg_sft/experiment/phase2_v8_release_gate.py",
    "src/eg_sft/experiment/phase2_v8_worker_lease.py",
    "src/eg_sft/experiment/budget_equivalent_audit_v4.py",
    "src/eg_sft/experiment/cell_evidence_package.py",
    "src/eg_sft/training/token_budget.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    for relative in sorted(PATHS):
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        if not path.is_file():
            raise ValueError(f"v8 semantic file is missing: {relative}")
        files.append({"path": relative, "sha256": file_sha256(path)})
    content = {
        "schema_version": "phase2-v8-semantic-code-manifest-v1",
        "protocol_id": "phase2-clean-common24-v8",
        "historical_parent_commit": "54a232d60cba939f0ea1f212e5c8aae2a73bbd3c",
        "files": files,
    }
    payload = content | {"manifest_content_sha256": canonical_json_sha256(content)}
    output = args.output.resolve()
    write_exclusive_or_verify(output, canonical_json_bytes(payload))
    print(json.dumps({"status": "PASS", "file_count": len(files), "sha256": file_sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
