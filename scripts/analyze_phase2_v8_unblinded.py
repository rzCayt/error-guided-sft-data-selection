"""CPU-only primary analysis after all 24 clean v8 cells are audited."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

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
from eg_sft.experiment.phase2_v8_statistics import (  # noqa: E402
    complete_leave_one_out,
    crossed_common_bootstrap,
    fixed_seed_common_bootstrap,
    seed_block_effects,
    validate_clean_cells,
)


CONFIRMATION = "PHASE2_V8_CLEAN24_UNBLIND_APPROVED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages-gpu0", type=Path, required=True)
    parser.add_argument("--packages-gpu1", type=Path, required=True)
    parser.add_argument("--progress-gate", type=Path, required=True)
    parser.add_argument("--precision-simulation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--operator-confirmation", required=True)
    args = parser.parse_args()
    if args.operator_confirmation != CONFIRMATION:
        raise ValueError("exact v8 unblinding confirmation is required")
    gate = read_json(args.progress_gate.resolve())
    if gate.get("unblinding_permitted") is not True or int(gate.get("complete_count", -1)) != 24:
        raise ValueError("v8 24-cell gate is closed")
    precision = read_json(args.precision_simulation.resolve())
    if precision.get("equivalence_status") != "EXPLORATORY_ONLY":
        raise ValueError("v8 precision boundary changed")
    cells, evidence = load_evidence_roots(
        [args.packages_gpu0.resolve(), args.packages_gpu1.resolve()]
    )
    validate_clean_cells(cells)
    accuracy = fixed_seed_common_bootstrap(
        cells=cells,
        metric="accuracy",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    strict = fixed_seed_common_bootstrap(
        cells=cells,
        metric="strict_parse_rate",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 1,
    )
    seed_resampled_accuracy = crossed_common_bootstrap(
        cells=cells,
        metric="accuracy",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 2,
    )
    seed_resampled_strict = crossed_common_bootstrap(
        cells=cells,
        metric="strict_parse_rate",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 3,
    )
    leave_one_out = complete_leave_one_out(cells=cells, metric="accuracy")
    seed_effects = seed_block_effects(cells=cells, metric="accuracy")
    cell_rows = [cell_metrics(cell) for cell in cells]
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    payloads = {
        "accuracy_fixed_seed_primary.json": accuracy,
        "strict_format_fixed_seed_primary.json": strict,
        "accuracy_seed_resampled_sensitivity.json": seed_resampled_accuracy,
        "strict_format_seed_resampled_sensitivity.json": seed_resampled_strict,
        "seed_block_effects.json": seed_effects,
        "complete_leave_one_out.json": leave_one_out,
        "method_summary.json": summarize_methods(cell_rows),
        "evidence_index.json": evidence,
    }
    for name, payload in payloads.items():
        write_exclusive_or_verify(output_dir / name, canonical_json_bytes(payload))
    write_exclusive_or_verify(
        output_dir / "cell_metrics.jsonl",
        b"".join(canonical_json_bytes(row) for row in cell_rows),
    )
    primary = accuracy["results"]["gsm8k"]
    claims = {
        "schema_version": "phase2-v8-claim-ledger-v1",
        "study_label": "preregistered clean-environment replication block",
        "primary_estimand": "rds_error_common_mix - random_common_mix",
        "primary_result": primary,
        "allowed_claim_class": primary["effect_diagnostic"],
        "primary_training_seed_role": "fixed_observed_blocks",
        "seed_population_inference": "exploratory_sensitivity_only",
        "equivalence_is_exploratory": True,
        "equivalence_must_not_be_used_as_primary_claim": True,
        "historical_seed17_in_primary": False,
        "historical_results_role": "external_replication_only",
        "free_mix_not_tested": True,
        "generalization_beyond_current_regime_not_supported": True,
    }
    write_exclusive_or_verify(
        output_dir / "claim_evidence_ledger.json", canonical_json_bytes(claims)
    )
    manifest = {
        "schema_version": "phase2-v8-analysis-manifest-v1",
        "status": "PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "audited_cell_count": 24,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "progress_gate_sha256": file_sha256(args.progress_gate.resolve()),
        "precision_simulation_sha256": file_sha256(args.precision_simulation.resolve()),
        "output_sha256": {path.name: file_sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
        "gpu_accessed": False,
    }
    write_exclusive_or_verify(
        output_dir / "analysis_manifest.json", canonical_json_bytes(manifest)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
