import json
from pathlib import Path

import pytest
import torch

from eg_sft.experiment.rds_full_pool import (
    build_b500_selection_manifest,
    build_score_rows,
    chunk_bounds,
    chunk_count,
    ordered_value_sha256,
    tensor_sha256,
    validate_chunk_manifest,
)


def test_chunk_bounds_cover_every_row_once() -> None:
    assert chunk_count(10_000, 128) == 79
    bounds = [
        chunk_bounds(10_000, 128, index)
        for index in range(chunk_count(10_000, 128))
    ]
    assert bounds[0] == (0, 128)
    assert bounds[-1] == (9984, 10_000)
    assert sum(end - start for start, end in bounds) == 10_000
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))


def test_chunk_manifest_binds_ids_artifact_and_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "chunk_0000_attempt.pt"
    artifact.write_bytes(b"artifact")
    artifact_sha256 = __import__("hashlib").sha256(b"artifact").hexdigest()
    ids = ["a", "b"]
    manifest = {
        "status": "COMPLETE",
        "kind": "candidate",
        "chunk_index": 0,
        "representation_version": "representation-v1",
        "run_contract_sha256": "contract",
        "row_count": 2,
        "ordered_id_sha256": ordered_value_sha256(ids),
        "artifact_file": artifact.name,
        "artifact_sha256": artifact_sha256,
    }
    validate_chunk_manifest(
        manifest=manifest,
        expected_kind="candidate",
        expected_chunk_index=0,
        expected_ids=ids,
        expected_representation_version="representation-v1",
        expected_run_contract_sha256="contract",
        artifact_path=artifact,
        artifact_sha256=artifact_sha256,
    )
    changed = dict(manifest)
    changed["ordered_id_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ordered ID"):
        validate_chunk_manifest(
            manifest=changed,
            expected_kind="candidate",
            expected_chunk_index=0,
            expected_ids=ids,
            expected_representation_version="representation-v1",
            expected_run_contract_sha256="contract",
            artifact_path=artifact,
            artifact_sha256=artifact_sha256,
        )


def _candidate(index: int) -> dict:
    return {
        "candidate_id": f"candidate-{index}",
        "source_dataset": "source",
        "source_id": f"source-{index}",
        "source_index": index,
        "prompt_sha256": f"prompt-{index}",
        "response_sha256": f"response-{index}",
        "user_prompt_sha256": f"user-{index}",
        "selection_priority_sha256": f"priority-{index}",
        "selection_rank": index,
        "eligible_index": index,
        "rds_text_sha256": f"text-{index}",
        "total_tokens": 20,
        "supervised_tokens": 5,
    }


def test_score_rows_use_frozen_all_and_error_query_groups() -> None:
    queries = [
        {"record_id": "q0", "is_error_query": False},
        {"record_id": "q1", "is_error_query": True},
    ]
    query_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    candidate_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ]
    )
    rows, metrics = build_score_rows(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        query_inventory=queries,
        eligible_candidates=[_candidate(index) for index in range(3)],
        selection_budget=2,
    )
    assert [row["all_query_rank"] for row in rows] == [0, 1, 2]
    assert [row["error_query_rank"] for row in rows] == [2, 0, 1]
    assert metrics["all_query_count"] == 2
    assert metrics["error_query_count"] == 1
    assert metrics["selection_budget"] == 2


def test_selection_manifests_follow_only_the_requested_frozen_rank() -> None:
    rows = []
    for index in range(3):
        rows.append(
            _candidate(index)
            | {
                "all_query_rank": index,
                "all_query_score": 1.0 - index / 2,
                "error_query_rank": 2 - index,
                "error_query_score": index / 2,
            }
        )
    provenance = {"candidate_scores_sha256": "a" * 64}
    all_manifest = build_b500_selection_manifest(
        strategy="rds_all",
        score_rows=rows,
        budget=2,
        selection_seed=20260722,
        scoring_provenance=provenance,
    )
    error_manifest = build_b500_selection_manifest(
        strategy="rds_error",
        score_rows=rows,
        budget=2,
        selection_seed=20260722,
        scoring_provenance=provenance,
    )
    assert [
        row["candidate_id"] for row in all_manifest["selected_candidates"]
    ] == ["candidate-0", "candidate-1"]
    assert [
        row["candidate_id"] for row in error_manifest["selected_candidates"]
    ] == ["candidate-2", "candidate-1"]
    assert all_manifest["selected_id_sha256"] != error_manifest["selected_id_sha256"]
    assert all_manifest["manifest_content_sha256"]
    json.dumps(all_manifest)


def test_tensor_hash_is_shape_and_value_sensitive() -> None:
    base = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert tensor_sha256(base) == tensor_sha256(base.clone())
    assert tensor_sha256(base) != tensor_sha256(base + 1)
    assert tensor_sha256(base) != tensor_sha256(base.reshape(-1))
    assert tensor_sha256(base) != tensor_sha256(base.to(torch.float16))
