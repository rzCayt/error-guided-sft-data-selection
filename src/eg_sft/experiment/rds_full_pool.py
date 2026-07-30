"""Pure helpers for resumable full-pool RDS+ scoring and B=500 selection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from eg_sft.selection.rds import (
    cosine_similarity_matrix,
    rank_scores,
    round_robin_order,
)
from eg_sft.training.b500 import selected_id_sha256


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_value_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    header = json.dumps(
        {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(array.tobytes())
    return digest.hexdigest()


def chunk_count(total: int, chunk_size: int) -> int:
    if total < 0:
        raise ValueError("total must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return (total + chunk_size - 1) // chunk_size


def chunk_bounds(total: int, chunk_size: int, chunk_index: int) -> tuple[int, int]:
    count = chunk_count(total, chunk_size)
    if chunk_index < 0 or chunk_index >= count:
        raise ValueError(
            f"chunk_index={chunk_index} is outside the valid range 0..{count - 1}"
        )
    start = chunk_index * chunk_size
    return start, min(total, start + chunk_size)


def chunk_manifest_filename(chunk_index: int) -> str:
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return f"chunk_{chunk_index:04d}.json"


def validate_chunk_manifest(
    *,
    manifest: Mapping[str, Any],
    expected_kind: str,
    expected_chunk_index: int,
    expected_ids: Sequence[str],
    expected_representation_version: str,
    expected_run_contract_sha256: str,
    artifact_path: Path,
    artifact_sha256: str,
) -> None:
    if manifest.get("status") != "COMPLETE":
        raise ValueError("chunk manifest is not complete")
    if manifest.get("kind") != expected_kind:
        raise ValueError("chunk kind changed")
    if int(manifest.get("chunk_index", -1)) != expected_chunk_index:
        raise ValueError("chunk index changed")
    if manifest.get("representation_version") != expected_representation_version:
        raise ValueError("representation version changed")
    if manifest.get("run_contract_sha256") != expected_run_contract_sha256:
        raise ValueError("run contract changed")
    if int(manifest.get("row_count", -1)) != len(expected_ids):
        raise ValueError("chunk row count changed")
    if manifest.get("ordered_id_sha256") != ordered_value_sha256(expected_ids):
        raise ValueError("chunk ordered ID hash changed")
    if Path(str(manifest.get("artifact_file", ""))).name != artifact_path.name:
        raise ValueError("chunk artifact filename changed")
    if manifest.get("artifact_sha256") != artifact_sha256:
        raise ValueError("chunk artifact hash changed")


def spearman_from_complete_orders(
    left: Sequence[int],
    right: Sequence[int],
) -> float:
    if len(left) != len(right) or set(left) != set(right):
        raise ValueError("orders must contain the same candidates")
    count = len(left)
    if count < 2:
        return 1.0
    left_rank = {candidate: rank for rank, candidate in enumerate(left)}
    right_rank = {candidate: rank for rank, candidate in enumerate(right)}
    squared_difference = sum(
        (left_rank[candidate] - right_rank[candidate]) ** 2
        for candidate in left_rank
    )
    return 1.0 - 6.0 * squared_difference / (count * (count**2 - 1))


def top_jaccard(
    left: Sequence[int],
    right: Sequence[int],
    count: int,
) -> float:
    if count <= 0 or count > len(left) or len(left) != len(right):
        raise ValueError("invalid top-set count")
    left_set = set(left[:count])
    right_set = set(right[:count])
    return len(left_set & right_set) / len(left_set | right_set)


def build_score_rows(
    *,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    query_inventory: Sequence[dict[str, Any]],
    eligible_candidates: Sequence[dict[str, Any]],
    selection_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if query_embeddings.shape[0] != len(query_inventory):
        raise ValueError("query embedding count changed")
    if candidate_embeddings.shape[0] != len(eligible_candidates):
        raise ValueError("candidate embedding count changed")
    if not query_inventory:
        raise ValueError("query inventory is empty")
    if selection_budget <= 0 or selection_budget > len(eligible_candidates):
        raise ValueError("selection budget is invalid")

    error_mask = torch.tensor(
        [bool(row["is_error_query"]) for row in query_inventory],
        dtype=torch.bool,
    )
    if int(error_mask.sum().item()) == 0:
        raise ValueError("error query group is empty")
    all_similarity = cosine_similarity_matrix(
        query_embeddings,
        candidate_embeddings,
    )
    error_similarity = cosine_similarity_matrix(
        query_embeddings[error_mask],
        candidate_embeddings,
    )
    all_order = round_robin_order(all_similarity)
    error_order = round_robin_order(error_similarity)
    all_scores = rank_scores(
        all_order,
        candidate_count=len(eligible_candidates),
    )
    error_scores = rank_scores(
        error_order,
        candidate_count=len(eligible_candidates),
    )
    all_ranks = {candidate: rank for rank, candidate in enumerate(all_order)}
    error_ranks = {
        candidate: rank for rank, candidate in enumerate(error_order)
    }

    score_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(eligible_candidates):
        score_rows.append(
            {
                **candidate,
                "all_query_rank": all_ranks[index],
                "all_query_score": all_scores[index],
                "error_query_rank": error_ranks[index],
                "error_query_score": error_scores[index],
            }
        )
    metrics = {
        "eligible_candidate_count": len(eligible_candidates),
        "all_query_count": len(query_inventory),
        "error_query_count": int(error_mask.sum().item()),
        "embedding_dimension": int(candidate_embeddings.shape[1]),
        "all_vs_error_order_identical": all_order == error_order,
        "all_vs_error_rank_spearman": spearman_from_complete_orders(
            all_order,
            error_order,
        ),
        "all_vs_error_top_budget_jaccard": top_jaccard(
            all_order,
            error_order,
            selection_budget,
        ),
        "selection_budget": selection_budget,
    }
    return score_rows, metrics


_SELECTION_FIELDS = (
    "candidate_id",
    "source_dataset",
    "source_id",
    "source_index",
    "prompt_sha256",
    "response_sha256",
    "user_prompt_sha256",
    "selection_priority_sha256",
    "selection_rank",
    "eligible_index",
    "rds_text_sha256",
    "total_tokens",
    "supervised_tokens",
    "all_query_rank",
    "all_query_score",
    "error_query_rank",
    "error_query_score",
)


def build_b500_selection_manifest(
    *,
    strategy: str,
    score_rows: Sequence[dict[str, Any]],
    budget: int,
    selection_seed: int,
    scoring_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    rank_fields = {
        "rds_all": "all_query_rank",
        "rds_error": "error_query_rank",
    }
    if strategy not in rank_fields:
        raise ValueError("unsupported RDS selection strategy")
    if budget <= 0 or budget > len(score_rows):
        raise ValueError("selection budget is invalid")
    ids = [str(row["candidate_id"]) for row in score_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate score rows contain duplicate IDs")
    if any(int(row.get("supervised_tokens", 0)) <= 0 for row in score_rows):
        raise ValueError("candidate score rows include an untrainable response")

    rank_field = rank_fields[strategy]
    ordered = sorted(
        score_rows,
        key=lambda row: (int(row[rank_field]), str(row["candidate_id"])),
    )
    ranks = [int(row[rank_field]) for row in ordered]
    if ranks != list(range(len(score_rows))):
        raise ValueError(f"{rank_field} must be a complete zero-based order")

    selected: list[dict[str, Any]] = []
    for rds_selection_rank, row in enumerate(ordered[:budget]):
        missing = [field for field in _SELECTION_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"{row['candidate_id']} is missing selection fields: {missing}"
            )
        selected.append(
            {
                field: row[field] for field in _SELECTION_FIELDS
            }
            | {
                "rds_selection_rank": rds_selection_rank,
                "rds_rank_field": rank_field,
            }
        )
    manifest = {
        "manifest_schema_version": "b500-rds-selection-v1",
        "strategy": strategy,
        "budget": budget,
        "selection_seed": selection_seed,
        "selection_rule": (
            f"ascending_{rank_field}_over_response_trainable_full_pool"
        ),
        "candidate_score_count": len(score_rows),
        "selected_candidates": selected,
        "selected_id_sha256": selected_id_sha256(selected),
        "selected_source_counts": dict(
            sorted(
                Counter(
                    str(row["source_dataset"]) for row in selected
                ).items()
            )
        ),
        "selected_total_tokens": sum(
            int(row["total_tokens"]) for row in selected
        ),
        "selected_supervised_tokens": sum(
            int(row["supervised_tokens"]) for row in selected
        ),
        "scoring_provenance": dict(scoring_provenance),
        "claim_boundary": (
            "This manifest freezes one B=500 training set. "
            "It is not a downstream training or evaluation result."
        ),
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest
