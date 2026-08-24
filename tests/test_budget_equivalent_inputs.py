import json
from pathlib import Path

import torch

from eg_sft.experiment.budget_equivalent_inputs import (
    cluster_near_duplicate_prompts,
    eligible_candidate_rows,
    export_similarity_artifact,
)
from eg_sft.training.b500 import file_sha256


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _chunk(directory: Path, kind: str, ids: list[str], values: torch.Tensor) -> None:
    directory.mkdir(parents=True)
    artifact = directory / "attempt.pt"
    torch.save(
        {
            "kind": kind,
            "chunk_index": 0,
            "run_contract_sha256": "a" * 64,
            "representation_version": "v1",
            "ids": ids,
            "embeddings": values,
        },
        artifact,
    )
    manifest = {
        "status": "COMPLETE",
        "chunk_index": 0,
        "start_index": 0,
        "artifact_file": artifact.name,
        "artifact_sha256": file_sha256(artifact),
    }
    (directory / "chunk_0000.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_eligible_candidate_rows_filters_full_audit_inventory_in_order() -> None:
    rows = [
        {"candidate_id": "c0", "response_only_trainable": True},
        {"candidate_id": "c1", "response_only_trainable": False},
        {"candidate_id": "c2", "response_only_trainable": True},
    ]
    assert [
        row["candidate_id"] for row in eligible_candidate_rows(rows)
    ] == ["c0", "c2"]


def test_similarity_export_binds_ids_and_values(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _chunk(
        run / "embedding_chunks" / "query",
        "query",
        ["q0", "q1"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    _chunk(
        run / "embedding_chunks" / "candidate",
        "candidate",
        ["c0", "c1"],
        torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
    )
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(queries, [{"record_id": "q0"}, {"record_id": "q1"}])
    _write_jsonl(candidates, [{"candidate_id": "c0"}, {"candidate_id": "c1"}])
    output = tmp_path / "similarity.pt"
    report = export_similarity_artifact(
        run_dir=run,
        query_inventory_path=queries,
        candidate_inventory_path=candidates,
        output_path=output,
    )
    payload = torch.load(output, map_location="cpu", weights_only=True)
    torch.testing.assert_close(payload["similarity"], torch.eye(2))
    assert report["shape"] == [2, 2]


def test_near_duplicate_clustering_groups_rewording_but_not_unrelated_text() -> None:
    rows, audit = cluster_near_duplicate_prompts(
        ["a", "b", "c"],
        [
            "compute the total number of red apples in the large basket today",
            "compute the total number of red apples in the large basket today please",
            "explain why the ocean has tides and how the moon contributes",
        ],
        ngram_size=3,
        containment_threshold=0.80,
    )
    clusters = {row["candidate_id"]: row["near_duplicate_cluster_id"] for row in rows}
    assert clusters["a"] == clusters["b"]
    assert clusters["a"] != clusters["c"]
    assert audit["multi_member_cluster_count"] == 1
