import pytest
import torch

from eg_sft.selection.budget_equivalent import (
    bootstrap_rds_priorities,
    build_common_mix_design,
    build_selection_manifest,
    median_pairwise_jaccard,
    solve_budget_equivalent_selection,
    stable_priority,
    stratified_query_bootstrap_indices,
)


def _candidates(count: int = 80) -> list[dict]:
    rows = []
    for index in range(count):
        response = 5 + index % 8
        prompt = 20 + index % 5
        rows.append(
            {
                "candidate_id": f"c{index:03d}",
                "source_dataset": "a" if index % 3 else "b",
                "supervised_tokens": response,
                "total_tokens": response + prompt,
                "user_prompt_sha256": f"prompt-{index}",
            }
        )
    return rows


def test_query_bootstrap_preserves_group_sizes_and_changes_with_seed() -> None:
    queries = [
        {"is_error_query": index >= 7, "record_id": f"q{index}"}
        for index in range(10)
    ]
    all_1, error_1 = stratified_query_bootstrap_indices(queries, seed=101)
    all_2, error_2 = stratified_query_bootstrap_indices(queries, seed=202)
    assert len(all_1) == 10
    assert len(error_1) == 3
    assert all(index >= 7 for index in error_1)
    assert (all_1, error_1) != (all_2, error_2)


def test_bootstrap_rds_priorities_are_complete_and_nonconstant() -> None:
    queries = [
        {"is_error_query": False, "record_id": "q0"},
        {"is_error_query": True, "record_id": "q1"},
    ]
    similarity = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.8, 0.7]])
    all_priorities, error_priorities, evidence = bootstrap_rds_priorities(
        similarity, queries, seed=101
    )
    assert sorted(all_priorities) == [0.0, 0.5, 1.0]
    assert sorted(error_priorities) == [0.0, 0.5, 1.0]
    assert evidence["all_bootstrap_count"] == 2
    assert evidence["error_bootstrap_count"] == 1


def test_common_mix_selection_meets_count_token_quota_and_is_deterministic() -> None:
    candidates = _candidates()
    design = build_common_mix_design(
        candidates,
        selection_count=20,
        target_response_tokens=170,
        requested_bin_count=4,
        minimum_source_quota=2,
        minimum_freedom_ratio=2.0,
    )
    priorities = [stable_priority(row["candidate_id"], 1101) for row in candidates]
    kwargs = {
        "selection_count": 20,
        "target_response_tokens": 170,
        "response_tolerance_fraction": 0.05,
        "common_design": design,
        "prompt_tolerance_fraction": 0.05,
        "total_tolerance_fraction": 0.05,
        "allow_exact_prompt_fallback": True,
    }
    selected_1, audit_1 = solve_budget_equivalent_selection(
        candidates, priorities, **kwargs
    )
    selected_2, audit_2 = solve_budget_equivalent_selection(
        candidates, priorities, **kwargs
    )
    assert [row["candidate_id"] for row in selected_1] == [
        row["candidate_id"] for row in selected_2
    ]
    assert audit_1["selected_count"] == 20
    assert audit_1["response_relative_error"] <= 0.05
    assert audit_1["common_mix_quota_matches"] is True
    assert audit_1["selected_id_sha256"] == audit_2["selected_id_sha256"]


def test_formal_selection_rejects_missing_near_duplicate_clusters() -> None:
    candidates = _candidates()
    priorities = [stable_priority(row["candidate_id"], 1101) for row in candidates]
    with pytest.raises(ValueError, match="near-duplicate cluster"):
        solve_budget_equivalent_selection(
            candidates,
            priorities,
            selection_count=20,
            target_response_tokens=170,
            response_tolerance_fraction=0.05,
            common_design=None,
            prompt_tolerance_fraction=0.05,
            total_tolerance_fraction=0.05,
        )


def test_manifest_and_jaccard_are_auditable() -> None:
    selected = _candidates(4)
    manifest = build_selection_manifest(
        method="random_free_mix",
        selection_seed=1101,
        train_seed=17,
        selected=selected,
        audit={"status": "PASS"},
        provenance={"candidate_inventory_sha256": "a" * 64},
    )
    assert manifest["budget"] == 4
    assert len(manifest["manifest_content_sha256"]) == 64
    assert median_pairwise_jaccard(
        [["a", "b"], ["b", "c"], ["a", "c"]]
    ) == 1 / 3
