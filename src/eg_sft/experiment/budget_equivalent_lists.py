"""Build immutable Phase 1 list manifests and selection-information gates."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from eg_sft.experiment.budget_equivalent_protocol import (
    preflight_protocol,
    read_json_object,
    repository_path,
    validate_protocol_config,
)
from eg_sft.selection.budget_equivalent import (
    CORE_METHODS,
    bootstrap_rds_priorities,
    build_common_mix_design,
    build_selection_manifest,
    canonical_json_sha256,
    jaccard,
    median_pairwise_jaccard,
    solve_budget_equivalent_selection,
    stable_priority,
)
from eg_sft.training.b500 import file_sha256, read_jsonl


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_similarity(
    path: Path,
    *,
    query_ids: list[str],
    candidate_ids: list[str],
) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("similarity artifact must be a mapping")
    similarity = payload.get("similarity")
    if not isinstance(similarity, torch.Tensor):
        raise ValueError("similarity artifact has no tensor named similarity")
    if list(payload.get("query_ids", [])) != query_ids:
        raise ValueError("similarity query order differs from frozen inventory")
    if list(payload.get("candidate_ids", [])) != candidate_ids:
        raise ValueError("similarity candidate order differs from frozen inventory")
    if tuple(similarity.shape) != (len(query_ids), len(candidate_ids)):
        raise ValueError("similarity tensor shape differs from frozen inventories")
    if not torch.isfinite(similarity).all():
        raise ValueError("similarity tensor contains non-finite values")
    return similarity.float().contiguous()


def _load_duplicate_clusters(path: Path) -> dict[str, str]:
    rows = read_jsonl(path)
    mapping: dict[str, str] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        cluster_id = str(row.get("near_duplicate_cluster_id", ""))
        if not candidate_id or not cluster_id:
            raise ValueError("near-duplicate cluster rows require candidate_id and cluster ID")
        if candidate_id in mapping:
            raise ValueError(f"duplicate cluster assignment for {candidate_id}")
        mapping[candidate_id] = cluster_id
    return mapping


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["candidate_id"]) for row in rows]


def _changed_fraction(left: list[str], right: list[str]) -> float:
    if len(left) != len(right):
        raise ValueError("changed-fraction lists must have equal length")
    return 1.0 - len(set(left) & set(right)) / len(left)


def build_phase1_lists(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path | None = None,
    engineering_allow_exact_prompt_fallback: bool = False,
) -> dict[str, Any]:
    config = read_json_object(config_path)
    validate_protocol_config(config)
    preflight = preflight_protocol(repo_root=repo_root, config_path=config_path)
    required_names = {"protocol_config", "candidate_inventory", "query_inventory", "similarity_artifact"}
    blocked = [
        name
        for name in required_names
        if preflight["bindings"][name]["status"] != "READY"
    ]
    cluster_ready = preflight["bindings"]["near_duplicate_clusters"]["status"] == "READY"
    if blocked:
        raise ValueError(f"selection inputs are not frozen and ready: {blocked}")
    if not cluster_ready and not engineering_allow_exact_prompt_fallback:
        raise ValueError("formal list build requires the frozen near-duplicate cluster manifest")

    candidates = read_jsonl(repository_path(repo_root, config["candidate_inventory"]["path"]))
    queries = read_jsonl(repository_path(repo_root, config["query_inventory"]["path"]))
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    query_ids = [str(row["record_id"]) for row in queries]
    similarity_path = repository_path(repo_root, config["similarity_artifact"]["path"])
    similarity = _load_similarity(
        similarity_path,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )
    clusters = None
    cluster_path = repository_path(repo_root, config["near_duplicate_clusters"]["path"])
    if cluster_ready:
        clusters = _load_duplicate_clusters(cluster_path)

    selection = config["selection"]
    design = build_common_mix_design(
        candidates,
        selection_count=int(selection["selected_example_count"]),
        target_response_tokens=int(selection["target_response_supervision_tokens"]),
        requested_bin_count=int(selection["requested_response_length_bins"]),
        minimum_source_quota=int(selection["minimum_source_quota"]),
        minimum_freedom_ratio=float(selection["minimum_freedom_ratio"]),
    )
    root = (
        output_root.resolve()
        if output_root is not None
        else repository_path(repo_root, config["output_root"])
    )
    root.mkdir(parents=True, exist_ok=False)
    provenance = {
        "protocol_config_sha256": file_sha256(config_path),
        "candidate_inventory_sha256": file_sha256(
            repository_path(repo_root, config["candidate_inventory"]["path"])
        ),
        "query_inventory_sha256": file_sha256(
            repository_path(repo_root, config["query_inventory"]["path"])
        ),
        "similarity_sha256": file_sha256(similarity_path),
        "near_duplicate_clusters_sha256": file_sha256(cluster_path) if cluster_ready else None,
        "formal_near_duplicate_control": cluster_ready,
    }
    _write_json_exclusive(root / "common_mix_design.json", dataclasses.asdict(design))

    method_index: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    error_id_sets: list[list[str]] = []
    rds_seeds = selection["selection_replicate_seeds"]
    random_seeds = selection["random_priority_seeds"]
    for replicate_index, (rds_seed, random_seed) in enumerate(
        zip(rds_seeds, random_seeds, strict=True), start=1
    ):
        all_priorities, error_priorities, bootstrap_evidence = bootstrap_rds_priorities(
            similarity,
            queries,
            seed=int(rds_seed),
        )
        random_priorities = [
            stable_priority(candidate_id, int(random_seed)) for candidate_id in candidate_ids
        ]
        priorities_by_method = {
            "random_free_mix": random_priorities,
            "rds_error_free_mix": error_priorities,
            "random_common_mix": random_priorities,
            "rds_error_common_mix": error_priorities,
        }
        selected_by_method: dict[str, list[dict[str, Any]]] = {}
        for method in CORE_METHODS:
            is_common = method.endswith("common_mix")
            chosen, audit = solve_budget_equivalent_selection(
                candidates,
                priorities_by_method[method],
                selection_count=int(selection["selected_example_count"]),
                target_response_tokens=int(selection["target_response_supervision_tokens"]),
                response_tolerance_fraction=float(selection["response_tolerance_fraction"]),
                common_design=design if is_common else None,
                prompt_tolerance_fraction=float(selection["common_prompt_tolerance_fraction"]),
                total_tolerance_fraction=float(selection["common_total_tolerance_fraction"]),
                duplicate_clusters=clusters,
                allow_exact_prompt_fallback=engineering_allow_exact_prompt_fallback,
            )
            selected_by_method[method] = chosen
            method_seed = int(random_seed) if method.startswith("random") else int(rds_seed)
            manifest = build_selection_manifest(
                method=method,
                selection_seed=method_seed,
                train_seed=int(selection["phase1_train_seed"]),
                selected=chosen,
                audit=audit,
                provenance=provenance | bootstrap_evidence,
            )
            relative = Path(f"replicate_{replicate_index:02d}") / method / "selection_manifest.json"
            path = root / relative
            _write_json_exclusive(path, manifest)
            method_index.append(
                {
                    "replicate_index": replicate_index,
                    "method": method,
                    "selection_seed": method_seed,
                    "train_seed": int(selection["phase1_train_seed"]),
                    "path": relative.as_posix(),
                    "sha256": file_sha256(path),
                    "selected_id_sha256": manifest["selected_id_sha256"],
                }
            )

        all_selected, all_audit = solve_budget_equivalent_selection(
            candidates,
            all_priorities,
            selection_count=int(selection["selected_example_count"]),
            target_response_tokens=int(selection["target_response_supervision_tokens"]),
            response_tolerance_fraction=float(selection["response_tolerance_fraction"]),
            common_design=None,
            prompt_tolerance_fraction=float(selection["common_prompt_tolerance_fraction"]),
            total_tolerance_fraction=float(selection["common_total_tolerance_fraction"]),
            duplicate_clusters=clusters,
            allow_exact_prompt_fallback=engineering_allow_exact_prompt_fallback,
        )
        all_ids = _ids(all_selected)
        error_ids = _ids(selected_by_method["rds_error_free_mix"])
        error_id_sets.append(error_ids)
        from scipy.stats import spearmanr

        rank_correlation = float(spearmanr(all_priorities, error_priorities).statistic)
        gate_row = {
            "replicate_index": replicate_index,
            "rds_seed": int(rds_seed),
            "random_seed": int(random_seed),
            "error_vs_all_top500_changed_fraction": _changed_fraction(
                error_ids, all_ids
            ),
            "error_vs_all_full_rank_spearman": rank_correlation,
            "rds_error_vs_random_free_jaccard": jaccard(
                error_ids, _ids(selected_by_method["random_free_mix"])
            ),
            "rds_error_vs_random_common_jaccard": jaccard(
                _ids(selected_by_method["rds_error_common_mix"]),
                _ids(selected_by_method["random_common_mix"]),
            ),
            "rds_all_gate_selected_id_sha256": all_audit["selected_id_sha256"],
        }
        gate_rows.append(gate_row)
        _write_json_exclusive(
            root / f"replicate_{replicate_index:02d}" / "information_gate.json",
            gate_row | bootstrap_evidence,
        )

    gates = config["information_gates"]
    stability = median_pairwise_jaccard(error_id_sets)
    changed_values = [row["error_vs_all_top500_changed_fraction"] for row in gate_rows]
    correlations = [row["error_vs_all_full_rank_spearman"] for row in gate_rows]
    minimum_freedom = min(
        design.stratum_candidate_counts[key] / quota
        for key, quota in design.stratum_quotas.items()
        if quota > 0
    )
    forced_fraction = design.forced_selected_count / int(selection["selected_example_count"])
    error_conditioning_gate = (
        min(changed_values)
        >= float(gates["minimum_error_vs_all_top500_changed_fraction"])
        and max(correlations) < float(gates["maximum_error_vs_all_rank_spearman"])
        and stability >= float(gates["minimum_error_selection_stability_jaccard"])
    )
    targeted_policy_gate = (
        all(row["rds_error_vs_random_free_jaccard"] < 1.0 for row in gate_rows)
        and all(row["rds_error_vs_random_common_jaccard"] < 1.0 for row in gate_rows)
        and minimum_freedom >= float(selection["minimum_freedom_ratio"])
        and forced_fraction <= float(selection["maximum_forced_selected_fraction"])
    )
    summary = {
        "schema_version": "budget-equivalent-information-gates-v3",
        "replicate_count": len(gate_rows),
        "replicates": gate_rows,
        "minimum_error_vs_all_top500_changed_fraction": min(changed_values),
        "maximum_error_vs_all_full_rank_spearman": max(correlations),
        "median_error_selection_stability_jaccard": stability,
        "minimum_common_stratum_freedom_ratio": minimum_freedom,
        "forced_selected_fraction": forced_fraction,
        "targeted_policy_gate_passed": targeted_policy_gate,
        "error_conditioning_increment_gate_passed": error_conditioning_gate,
        "rds_all_full_training_permitted": error_conditioning_gate,
        "phase1_core_matrix_permitted": targeted_policy_gate and cluster_ready,
        "formal_near_duplicate_control": cluster_ready,
        "claim_boundary": (
            "Failure of the error-conditioning gate forbids an incremental-error claim; "
            "it does not by itself invalidate the prespecified RDS-error policy versus random."
        ),
    }
    _write_json_exclusive(root / "information_gates.json", summary)
    index = {
        "schema_version": "budget-equivalent-phase1-selection-index-v3",
        "protocol_config_sha256": file_sha256(config_path),
        "selection_count": len(method_index),
        "expected_selection_count": 16,
        "selections": method_index,
        "information_gates_sha256": file_sha256(root / "information_gates.json"),
        "common_mix_design_sha256": file_sha256(root / "common_mix_design.json"),
        "formal_phase1_ready": summary["phase1_core_matrix_permitted"],
    }
    index["index_content_sha256"] = canonical_json_sha256(index)
    _write_json_exclusive(root / "selection_index.json", index)
    return index
