"""CPU-only 48-cell analysis, available only after the explicit full gate."""

from __future__ import annotations

import argparse
import json
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
from eg_sft.experiment.budget_equivalent_phase1_analysis import (  # noqa: E402
    cell_metrics,
    summarize_methods,
)
from eg_sft.experiment.phase2_v7_evidence import load_evidence_roots  # noqa: E402
from eg_sft.experiment.phase2_v7_statistics import (  # noqa: E402
    TASKS,
    descriptive_variance_components,
    hierarchical_four_method_bootstrap,
    validate_confirmatory_cells,
)


CONFIRMATION = "PHASE2_V7_48CELL_UNBLIND_APPROVED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-package-root", type=Path, required=True)
    parser.add_argument("--new-package-root-gpu0", type=Path, required=True)
    parser.add_argument("--new-package-root-gpu1", type=Path, required=True)
    parser.add_argument("--parent-gate", type=Path, required=True)
    parser.add_argument("--phase2-progress-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--operator-confirmation", required=True)
    args = parser.parse_args()
    if args.operator_confirmation != CONFIRMATION:
        raise ValueError("exact 48-cell unblinding confirmation is required")
    parent_gate = read_json(args.parent_gate.resolve())
    progress = read_json(args.phase2_progress_gate.resolve())
    if (
        int(parent_gate.get("formal_audited_pass_count", -1)) != 16
        or int(parent_gate.get("ood_audited_pass_count", -1)) != 16
        or parent_gate.get("unblinding_permitted") is not True
    ):
        raise ValueError("parent 16-cell audit gate is closed")
    if (
        int(progress.get("new_complete_count", -1)) != 32
        or progress.get("unblinding_permitted") is not True
    ):
        raise ValueError("Phase-2 32-cell audit gate is closed")
    cells, evidence = load_evidence_roots(
        [
            args.parent_package_root,
            args.new_package_root_gpu0,
            args.new_package_root_gpu1,
        ]
    )
    validate_confirmatory_cells(cells)
    accuracy = hierarchical_four_method_bootstrap(
        cells=cells,
        metric="accuracy",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    strict = hierarchical_four_method_bootstrap(
        cells=cells,
        metric="strict_parse_rate",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 1,
    )
    variance = {
        task: descriptive_variance_components(
            cells=cells, task=task, metric="accuracy"
        )
        for task in TASKS
    }
    cell_rows = [cell_metrics(cell) for cell in cells]
    method_summary = summarize_methods(cell_rows)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "accuracy_hierarchical_bootstrap.json": accuracy,
        "strict_format_hierarchical_bootstrap.json": strict,
        "descriptive_variance_components.json": variance,
        "method_summary.json": method_summary,
        "evidence_index.json": evidence,
    }
    for name, payload in outputs.items():
        write_exclusive_or_verify(output_dir / name, canonical_json_bytes(payload))
    cell_bytes = b"".join(canonical_json_bytes(row) for row in cell_rows)
    write_exclusive_or_verify(output_dir / "cell_metrics.jsonl", cell_bytes)
    primary = accuracy["results"]["gsm8k"]["common_rds_minus_random"]
    claim_ledger = {
        "schema_version": "phase2-v7-claim-ledger-v1",
        "primary_estimand": "rds_error_common_mix - random_common_mix",
        "primary_metric": "GSM8K exact numeric accuracy",
        "primary_result": primary,
        "allowed_primary_claim_class": primary["threshold_diagnostic"],
        "free_mix_is_secondary": True,
        "interaction_is_secondary": True,
        "error_conditioning_incremental_claim_permanently_abandoned": True,
        "random_and_rds_list_indices_not_treated_as_paired": True,
        "training_seed_is_a_fixed_block_in_model_sensitivity": True,
        "no_general_reasoning_claim_without_cross_model_and_budget_replication": True,
    }
    write_exclusive_or_verify(
        output_dir / "claim_evidence_ledger.json",
        canonical_json_bytes(claim_ledger),
    )
    manifest = {
        "schema_version": "phase2-v7-48cell-analysis-manifest-v1",
        "status": "PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "audited_cell_count": len(cells),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "parent_gate_sha256": file_sha256(args.parent_gate.resolve()),
        "phase2_progress_gate_sha256": file_sha256(
            args.phase2_progress_gate.resolve()
        ),
        "output_sha256": {
            path.name: file_sha256(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
        "gpu_accessed": False,
    }
    write_exclusive_or_verify(
        output_dir / "analysis_manifest.json", canonical_json_bytes(manifest)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
