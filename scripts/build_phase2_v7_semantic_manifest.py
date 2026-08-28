"""Build the frozen semantic-code hash manifest before any GPU starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v7_environment import (  # noqa: E402
    semantic_code_manifest,
)


DEFAULT_PATHS = (
    "configs/phase2_crossed_48cell_v7.json",
    "configs/phase2_v7_environment_contract.json",
    "configs/phase2_v7_legacy_batch1_contract.json",
    "configs/phase2_v7_stop_go_rules.json",
    "scripts/run_identifiable_budget_v4_cell.py",
    "scripts/run_budget_equivalent_cell.py",
    "scripts/run_budget_equivalent_cell_v3.py",
    "scripts/run_budget_equivalent_eval_worker.py",
    "scripts/run_cloud_v2_formal_eval_worker.py",
    "scripts/run_budget_equivalent_ood_eval_worker.py",
    "scripts/audit_budget_equivalent_cell_v5.py",
    "scripts/audit_budget_equivalent_ood_v3.py",
    "scripts/preflight_phase2_v7.py",
    "scripts/collect_phase2_v7_environment.py",
    "scripts/prepare_phase2_v7_static_runtime.py",
    "scripts/run_phase2_v7_canary.py",
    "scripts/finalize_phase2_v7_qualification.py",
    "scripts/run_phase2_v7_worker.py",
    "scripts/aggregate_phase2_v7_progress.py",
    "scripts/analyze_phase2_v7_unblinded.py",
    "src/eg_sft/evaluation/identifiable_batch_backend.py",
    "src/eg_sft/evaluation/phase2_v7_canary.py",
    "src/eg_sft/experiment/phase2_crossed_v7.py",
    "src/eg_sft/experiment/phase2_v7_control.py",
    "src/eg_sft/experiment/phase2_v7_environment.py",
    "src/eg_sft/experiment/phase2_v7_statistics.py",
    "src/eg_sft/experiment/phase2_v7_evidence.py",
    "src/eg_sft/experiment/budget_equivalent_audit_v4.py",
    "src/eg_sft/training/token_budget.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2_v7_preflight/semantic_code_manifest.json"),
    )
    args = parser.parse_args()
    payload = semantic_code_manifest(
        root=ROOT, paths=DEFAULT_PATHS, parent_commit=args.parent_commit
    )
    write_exclusive_or_verify(args.output.resolve(), canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
