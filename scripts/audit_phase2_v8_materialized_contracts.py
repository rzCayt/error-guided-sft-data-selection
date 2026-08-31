"""Read-only audit for all 24 materialized v8 training-input contracts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


SEED_INVARIANT_FIELDS = (
    "selection_manifest_sha256",
    "selected_count",
    "selected_id_set_sha256",
    "selected_id_order_sha256",
    "tokenized_input_sha256",
    "label_mask_sha256",
    "training_config_sha256",
    "optimizer_steps",
    "response_supervision_exposure_tokens",
    "prompt_token_count",
    "non_padding_token_count",
)

SEED_DERIVED_FIELDS = (
    "ordered_sample_occurrence_sha256",
    "optimizer_step_plan_sha256",
    "step_response_token_counts_sha256",
    "rng_map_sha256",
)


def audit_materialized_contracts(
    *, repo_root: Path, config_path: Path, contract_root: Path
) -> dict:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    contract_root = contract_root.resolve()
    matrix = read_json(config_path)
    validate_clean_common_matrix(matrix)
    completion_path = contract_root / "MATERIALIZATION_COMPLETE.json"
    completion = read_json(completion_path)
    checks: dict[str, bool] = {
        "completion_pass": completion.get("status") == "PASS",
        "config_hash_exact": completion.get("config_sha256")
        == file_sha256(config_path),
        "cell_count_24": completion.get("cell_count") == 24,
        "tokenizer_is_qwen2": str(completion.get("tokenizer_class", "")).startswith(
            "Qwen2"
        ),
        "mistral_regex_fix_not_applied": completion.get(
            "mistral_regex_fix_applied"
        )
        is False,
    }
    tokenizer_rows = completion.get("tokenizer_files", [])
    checks["tokenizer_manifest_complete"] = {
        str(row.get("name")) for row in tokenizer_rows
    } == {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}

    summary_by_cell = {
        str(row["cell_id"]): row for row in completion.get("cells", [])
    }
    checks["completion_lists_24_unique_cells"] = len(summary_by_cell) == 24
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    cell_rows = []
    for job in matrix["job_order"]:
        cell_id = str(job["cell_id"])
        cell_root = contract_root / cell_id
        hashes_path = cell_root / "training_input_hashes.json"
        diff_path = cell_root / "contract_diff.json"
        hashes = read_json(hashes_path)
        diff = read_json(diff_path)
        summary = summary_by_cell.get(cell_id, {})
        row_checks = {
            "hash_status_pass": hashes.get("status") == "PASS",
            "diff_status_pass": diff.get("status") == "PASS",
            "cell_id_exact": hashes.get("cell_id") == cell_id,
            "train_seed_exact": int(hashes.get("train_seed", -1))
            == int(job["train_seed"]),
            "selected_count_500": hashes.get("selected_count") == 500,
            "optimizer_steps_64": hashes.get("optimizer_steps") == 64,
            "positive_exposure": int(
                hashes.get("response_supervision_exposure_tokens", 0)
            )
            > 0,
            "hash_file_bound": summary.get("training_input_hashes_sha256")
            == file_sha256(hashes_path),
            "diff_file_bound": summary.get("contract_diff_sha256")
            == file_sha256(diff_path),
            "only_seed_declared_changed": diff.get("unexpected_scientific_changes")
            == [],
        }
        cell_rows.append(
            {
                "cell_id": cell_id,
                "status": "PASS" if all(row_checks.values()) else "FAIL",
                "checks": row_checks,
            }
        )
        groups[(str(job["method"]), int(job["replicate_index"]))].append(hashes)

    group_rows = []
    for (method, replicate), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: int(row["train_seed"]))
        invariant_checks = {
            field: len({json.dumps(row[field], sort_keys=True) for row in rows}) == 1
            for field in SEED_INVARIANT_FIELDS
        }
        derived_checks = {
            field: len({str(row[field]) for row in rows}) == 3
            for field in SEED_DERIVED_FIELDS
        }
        group_checks = {
            "three_expected_seeds": [int(row["train_seed"]) for row in rows]
            == [17, 29, 41],
            "all_seed_invariants_exact": all(invariant_checks.values()),
            "all_seed_derived_hashes_distinct": all(derived_checks.values()),
        }
        group_rows.append(
            {
                "method": method,
                "replicate_index": replicate,
                "status": "PASS" if all(group_checks.values()) else "FAIL",
                "checks": group_checks,
                "seed_invariant_fields": invariant_checks,
                "seed_derived_fields": derived_checks,
            }
        )

    checks["all_cells_pass"] = all(row["status"] == "PASS" for row in cell_rows)
    checks["eight_method_list_groups"] = len(group_rows) == 8
    checks["all_seed_groups_pass"] = all(
        row["status"] == "PASS" for row in group_rows
    )
    return {
        "schema_version": "phase2-v8-materialized-contract-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cell_rows": cell_rows,
        "seed_group_rows": group_rows,
        "artifact_hashes": {
            "matrix": file_sha256(config_path),
            "materialization_complete": file_sha256(completion_path),
        },
        "gpu_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_clean_common24_v8_canonical.json"),
    )
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_materialized_contracts(
        repo_root=ROOT,
        config_path=args.config,
        contract_root=args.contract_root,
    )
    write_exclusive_or_verify(args.output.resolve(), canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError("v8 materialized contract audit failed")


if __name__ == "__main__":
    main()
