"""Build one immutable, accuracy-blind Phase 1 policy-contrast audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_lists import _load_similarity  # noqa: E402
from eg_sft.experiment.budget_equivalent_inputs import eligible_candidate_rows  # noqa: E402
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    read_json_object,
    resolve_frozen_file,
)
from eg_sft.experiment.budget_equivalent_policy_contrast import (  # noqa: E402
    common_stratum_contrast,
    pair_policy_contrast,
)
from eg_sft.experiment.budget_equivalent_protocol import repository_path  # noqa: E402
from eg_sft.selection.budget_equivalent import (  # noqa: E402
    bootstrap_rds_priorities,
    canonical_json_sha256,
    median_pairwise_jaccard,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"selection manifest hash changed: {path}")
    manifest = read_json_object(path)
    claimed = manifest.get("manifest_content_sha256")
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    if claimed != canonical_json_sha256(content):
        raise ValueError(f"selection manifest self-hash changed: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/budget_equivalent_v3_protocol_frozen_20260824.json"),
    )
    parser.add_argument(
        "--selection-index",
        type=Path,
        default=Path(".aris/compute/budget_equivalent_v3_selections/selection_index.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol_config.resolve()
    protocol = read_json_object(protocol_path)
    selection_root = repository_path(ROOT, str(protocol["output_root"]))
    index_path = args.selection_index.resolve()
    index = read_json_object(index_path)
    if index.get("selection_count") != 16 or index.get("formal_phase1_ready") is not True:
        raise ValueError("selection index is not the frozen 16-list Phase 1 index")
    if index.get("protocol_config_sha256") != file_sha256(protocol_path):
        raise ValueError("selection index is not bound to the frozen protocol")

    candidate_path = resolve_frozen_file(
        repo_root=ROOT,
        binding=protocol["candidate_inventory"],
        label="candidate inventory",
    )
    query_path = resolve_frozen_file(
        repo_root=ROOT,
        binding=protocol["query_inventory"],
        label="query inventory",
    )
    similarity_path = resolve_frozen_file(
        repo_root=ROOT,
        binding=protocol["similarity_artifact"],
        label="similarity artifact",
    )
    candidates = eligible_candidate_rows(read_jsonl(candidate_path))
    queries = read_jsonl(query_path)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    query_ids = [str(row["record_id"]) for row in queries]
    similarity = _load_similarity(
        similarity_path, query_ids=query_ids, candidate_ids=candidate_ids
    )
    common_design = read_json_object(selection_root / "common_mix_design.json")

    manifests: dict[tuple[int, str], dict[str, Any]] = {}
    for binding in index["selections"]:
        key = (int(binding["replicate_index"]), str(binding["method"]))
        path = selection_root / str(binding["path"])
        manifests[key] = _load_manifest(path, expected_sha256=str(binding["sha256"]))
    if len(manifests) != 16:
        raise ValueError("selection index contains duplicate method/replicate identities")

    replicates = []
    method_id_sets: dict[str, list[list[str]]] = {}
    for replicate_index, seed in enumerate(
        protocol["selection"]["selection_replicate_seeds"], start=1
    ):
        _, error_priorities, bootstrap_evidence = bootstrap_rds_priorities(
            similarity, queries, seed=int(seed)
        )
        score_by_id = dict(zip(candidate_ids, error_priorities, strict=True))
        pair_reports = {}
        for mix in ("free_mix", "common_mix"):
            rds_method = f"rds_error_{mix}"
            random_method = f"random_{mix}"
            rds_rows = manifests[(replicate_index, rds_method)]["selected_candidates"]
            random_rows = manifests[(replicate_index, random_method)]["selected_candidates"]
            report = pair_policy_contrast(
                rds_rows=rds_rows,
                random_rows=random_rows,
                rds_priority_by_id=score_by_id,
            )
            if mix == "common_mix":
                report["within_stratum"] = common_stratum_contrast(
                    rds_rows=rds_rows,
                    random_rows=random_rows,
                    rds_priority_by_id=score_by_id,
                    stratum_candidate_counts=common_design["stratum_candidate_counts"],
                )
            pair_reports[mix] = report
        for method in ("random_free_mix", "rds_error_free_mix", "random_common_mix", "rds_error_common_mix"):
            rows = manifests[(replicate_index, method)]["selected_candidates"]
            method_id_sets.setdefault(method, []).append(
                [str(row["candidate_id"]) for row in rows]
            )
        replicates.append(
            {
                "replicate_index": replicate_index,
                "selection_seed": int(seed),
                "bootstrap_evidence": bootstrap_evidence,
                "pairs": pair_reports,
            }
        )

    method_stability = {
        method: median_pairwise_jaccard(id_sets)
        for method, id_sets in sorted(method_id_sets.items())
    }
    summary = {}
    for mix in ("free_mix", "common_mix"):
        rows = [row["pairs"][mix] for row in replicates]
        summary[mix] = {
            "mean_selected_id_jaccard": sum(row["selected_id_jaccard"] for row in rows)
            / len(rows),
            "minimum_replacement_fraction_of_budget": min(
                row["replacement_fraction_of_budget"] for row in rows
            ),
            "mean_rds_rank_percentile_lift": sum(
                row["mean_rds_rank_percentile_lift"] for row in rows
            )
            / len(rows),
            "all_replicates_nonidentical": all(
                row["contrast_is_nonidentical"] for row in rows
            ),
            "all_replicates_positive_rds_score_direction": all(
                row["rds_score_direction_is_positive"] for row in rows
            ),
        }
    output = {
        "schema_version": "budget-equivalent-policy-contrast-audit-v1",
        "status": "PASS",
        "accuracy_accessed": False,
        "downstream_results_accessed": False,
        "protocol_config_sha256": file_sha256(protocol_path),
        "selection_index_sha256": file_sha256(index_path),
        "candidate_inventory_sha256": file_sha256(candidate_path),
        "query_inventory_sha256": file_sha256(query_path),
        "similarity_artifact_sha256": file_sha256(similarity_path),
        "replicates": replicates,
        "method_selection_stability_median_pairwise_jaccard": method_stability,
        "summary": summary,
        "interpretation_boundary": (
            "This descriptive audit tests whether RDS and random create distinct policies "
            "with different RDS-score distributions. It does not measure downstream utility "
            "and does not add or change a Phase 1 gate."
        ),
        "truncation_rate_status": (
            "not_recoverable_from_frozen_candidate_inventory; selected token totals are "
            "reported, but no post-hoc truncation claim is made"
        ),
    }
    _write_json_exclusive(args.output.resolve(), output)
    args.output.resolve().with_suffix(".sha256").write_text(
        f"{file_sha256(args.output.resolve())}  {args.output.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "budget_equivalent_policy_contrast_audit",
                "output_sha256": file_sha256(args.output.resolve()),
                "accuracy_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
