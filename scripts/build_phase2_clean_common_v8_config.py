"""Mechanically generate the clean 24-cell common-mix v8 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_matrix import read_json_object  # noqa: E402
from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.experiment.phase2_clean_common_v8 import (  # noqa: E402
    validate_clean_common_matrix,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


SEQUENCE = (
    (1, 17), (2, 29), (3, 41), (4, 17),
    (1, 29), (2, 41), (3, 17), (4, 29),
    (1, 41), (2, 17), (3, 29), (4, 41),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json"),
    )
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--stop-go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_path = args.parent.resolve()
    parent = read_json_object(parent_path)
    parent_jobs = {row["cell_id"]: row for row in parent["job_order"]}
    jobs = []
    by_factor = {}
    for replicate in range(1, 5):
        for method in ("random_common_mix", "rds_error_common_mix"):
            parent_id = f"rep{replicate}_{method}_train17"
            source = parent_jobs[parent_id]
            for seed in (17, 29, 41):
                cell_id = f"v8_rep{replicate}_{method}_train{seed}"
                row = {
                    "cell_id": cell_id,
                    "parent_cell_id": parent_id,
                    "study": "clean_new_environment_common_block",
                    "method": method,
                    "selector": "random" if method.startswith("random_") else "rds_error",
                    "mix": "common_mix",
                    "replicate_index": replicate,
                    "train_seed": seed,
                    "selection_seed": int(source["selection_seed"]),
                    "parent_selection_manifest_sha256": source["selection_manifest"]["sha256"],
                }
                jobs.append(row)
                by_factor[(replicate, seed, row["selector"])] = cell_id
    waves = []
    for wave, (replicate, seed) in enumerate(SEQUENCE, start=1):
        random_cell = by_factor[(replicate, seed, "random")]
        rds_cell = by_factor[(replicate, seed, "rds_error")]
        gpu0, gpu1 = (
            (random_cell, rds_cell) if wave % 2 else (rds_cell, random_cell)
        )
        waves.append(
            {
                "wave": wave,
                "mix": "common_mix",
                "replicate_index": replicate,
                "train_seed": seed,
                "gpu0": gpu0,
                "gpu1": gpu1,
            }
        )

    def binding(path: Path) -> dict:
        value = path.resolve()
        value.relative_to(ROOT)
        return {"path": value.relative_to(ROOT).as_posix(), "sha256": file_sha256(value)}

    payload = {
        "matrix_version": "phase2-clean-common24-v8",
        "protocol_id": "phase2_clean_common24_v8",
        "parent_matrix": binding(parent_path),
        "output_root": ".aris/compute/phase2_clean_common24_v8_runs",
        "historical_external_cells": 8,
        "new_clean_cells": 24,
        "historical_cells_in_primary_analysis": False,
        "execution_policy": {
            "accuracy_blind_until_all_audits": True,
            "one_cell_per_gpu": True,
            "one_gpu_process_at_a_time": True,
            "ddp_forbidden": True,
            "formal_and_ood_audit_required": True,
            "automatic_unblinding": False,
            "required_new_audited_cells": 24,
            "historical_seed17_external_only": True,
            "free_mix_not_in_this_block": True,
        },
        "runtime_contracts": {
            "statistics": binding(args.statistics),
            "training_anchor": binding(args.anchor),
            "canary": binding(args.canary),
            "stop_go": binding(args.stop_go),
        },
        "analysis": {
            "primary_estimand": "rds_error_common_mix - random_common_mix",
            "primary_metric": "GSM8K exact numeric accuracy",
            "study_label": "preregistered clean-environment replication block",
            "training_seeds": [17, 29, 41],
            "selection_lists": [1, 2, 3, 4],
            "historical_seed17_role": "external_replication_only",
        },
        "job_order": jobs,
        "dual_gpu_schedule": waves,
    }
    validate_clean_common_matrix(payload)
    write_bytes_exclusive_or_verify(args.output.resolve(), canonical_json_bytes(payload))
    print(json.dumps({"status": "PASS", "jobs": 24, "waves": 12, "sha256": file_sha256(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
