"""Run the preregistered formal H1a statistics on frozen scoring and utility runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.h1a_analysis import (  # noqa: E402
    fixed_count_permutations,
    partial_spearman,
    top_bottom_mean_difference,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.selection.query_groups import load_jsonl  # noqa: E402
from eg_sft.selection.rds import (  # noqa: E402
    cosine_similarity_matrix,
    rank_scores,
    round_robin_order,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scores_from_similarity(similarity: torch.Tensor) -> list[float]:
    order = round_robin_order(similarity)
    return rank_scores(order, candidate_count=similarity.shape[1])


def _assert_close_scores(
    *,
    name: str,
    recomputed: list[float],
    stored: list[float],
) -> None:
    if len(recomputed) != len(stored):
        raise ValueError(f"{name} score lengths differ")
    maximum_difference = max(
        abs(left - right)
        for left, right in zip(recomputed, stored, strict=True)
    )
    if maximum_difference > 1e-12:
        raise ValueError(
            f"{name} stored scores do not match embeddings; "
            f"maximum difference={maximum_difference}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-run-dir", type=Path, required=True)
    parser.add_argument("--utility-run-dir", type=Path, required=True)
    parser.add_argument("--query-groups-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    if args.permutations != 1000:
        raise ValueError("formal H1a requires exactly 1000 permutations")

    scoring_dir = args.scoring_run_dir.resolve()
    utility_dir = args.utility_run_dir.resolve()
    query_dir = args.query_groups_dir.resolve()
    scoring_manifest = _read_json(scoring_dir / "manifest.json")
    utility_manifest = _read_json(utility_dir / "manifest.json")
    scoring_rows = load_jsonl(scoring_dir / "candidate_scores.jsonl")
    utility_rows = load_jsonl(utility_dir / "utility_measurements.jsonl")
    all_queries = load_jsonl(query_dir / "all_queries.jsonl")
    error_queries = load_jsonl(query_dir / "error_queries.jsonl")

    if len(scoring_rows) != 96:
        raise ValueError(f"expected 96 scoring rows, found {len(scoring_rows)}")
    if len(utility_rows) != 96:
        raise ValueError(f"expected 96 utility rows, found {len(utility_rows)}")
    if len(all_queries) != 448:
        raise ValueError(f"expected 448 diagnostic queries, found {len(all_queries)}")
    if len(error_queries) != 99:
        raise ValueError(f"expected 99 error queries, found {len(error_queries)}")
    if any(row.get("repeat_seed") != 17 for row in utility_rows):
        raise ValueError("formal utility rows must all use seed 17")
    if any(not row.get("response_only_trainable") for row in scoring_rows):
        raise ValueError("formal scoring rows must all be response-only trainable")

    candidate_ids = [row["candidate_id"] for row in scoring_rows]
    utility_by_id = {row["candidate_id"]: row for row in utility_rows}
    if len(set(candidate_ids)) != 96 or len(utility_by_id) != 96:
        raise ValueError("candidate IDs must be unique in both inputs")
    if set(candidate_ids) != set(utility_by_id):
        raise ValueError("scoring and utility candidate IDs differ")

    embeddings_path = scoring_dir / "embeddings.pt"
    embeddings = torch.load(
        embeddings_path,
        map_location="cpu",
        weights_only=True,
    )
    if embeddings["candidate_ids"] != candidate_ids:
        raise ValueError("embedding candidate order differs from score rows")
    query_ids = [row["record_id"] for row in all_queries]
    if embeddings["query_ids"] != query_ids:
        raise ValueError("embedding query order differs from all-query rows")

    error_ids = {row["record_id"] for row in error_queries}
    labeled_error_ids = {
        row["record_id"] for row in all_queries if not row["numeric_correct"]
    }
    if error_ids != labeled_error_ids:
        raise ValueError("error-query file differs from frozen numeric labels")
    error_indices = [
        index for index, query_id in enumerate(query_ids) if query_id in error_ids
    ]

    similarity = cosine_similarity_matrix(
        embeddings["query_embeddings"],
        embeddings["candidate_embeddings"],
    )
    all_scores = _scores_from_similarity(similarity)
    error_scores = _scores_from_similarity(similarity[error_indices])
    stored_all_scores = [float(row["all_query_score"]) for row in scoring_rows]
    stored_error_scores = [float(row["error_query_score"]) for row in scoring_rows]
    _assert_close_scores(
        name="all-query",
        recomputed=all_scores,
        stored=stored_all_scores,
    )
    _assert_close_scores(
        name="error-query",
        recomputed=error_scores,
        stored=stored_error_scores,
    )

    utilities = [
        float(utility_by_id[candidate_id]["utility"])
        for candidate_id in candidate_ids
    ]
    token_lengths = [
        float(row["training_total_tokens"]) for row in scoring_rows
    ]
    observed_rho = partial_spearman(
        predictor=error_scores,
        outcome=utilities,
        controls=[all_scores, token_lengths],
    )

    permutation_indices = fixed_count_permutations(
        item_count=len(all_queries),
        selected_count=len(error_indices),
        permutation_count=args.permutations,
        seed=args.seed,
    )
    permutation_rows: list[dict[str, Any]] = []
    permutation_statistics: list[float] = []
    for permutation_index, selected_indices in enumerate(permutation_indices):
        permuted_scores = _scores_from_similarity(similarity[selected_indices])
        statistic = partial_spearman(
            predictor=permuted_scores,
            outcome=utilities,
            controls=[all_scores, token_lengths],
        )
        if not math.isfinite(statistic):
            raise RuntimeError(
                f"non-finite statistic for permutation {permutation_index}"
            )
        permutation_statistics.append(statistic)
        permutation_rows.append(
            {
                "permutation_index": permutation_index,
                "selected_query_count": len(selected_indices),
                "partial_spearman": statistic,
            }
        )

    greater_or_equal_count = sum(
        statistic >= observed_rho for statistic in permutation_statistics
    )
    permutation_p = (1 + greater_or_equal_count) / (args.permutations + 1)
    top_bottom_difference, top_indices, bottom_indices = (
        top_bottom_mean_difference(
            scores=error_scores,
            utilities=utilities,
            group_count=24,
        )
    )
    top_set = set(top_indices)
    bottom_set = set(bottom_indices)
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "all_query_score": all_scores[index],
            "error_query_score": error_scores[index],
            "training_total_tokens": int(token_lengths[index]),
            "utility": utilities[index],
            "score_group": (
                "top_24"
                if index in top_set
                else "bottom_24"
                if index in bottom_set
                else "middle_48"
            ),
        }
        for index, candidate_id in enumerate(candidate_ids)
    ]

    rho_gate = observed_rho >= 0.15
    permutation_gate = permutation_p <= 0.10
    direction_gate = top_bottom_difference > 0
    h1a_passed = rho_gate and permutation_gate and direction_gate
    run_config = {
        "candidate_count": 96,
        "query_count": len(all_queries),
        "error_query_count": len(error_indices),
        "utility_seed": 17,
        "controls": ["all_query_score", "training_total_tokens"],
        "statistic": "partial_spearman_of_rank_residuals",
        "permutations": args.permutations,
        "permutation_rule": "sample_99_of_448_without_replacement",
        "permutation_p_value": "one_sided_plus_one_correction",
        "top_bottom_group_count": 24,
        "gates": {
            "partial_spearman_at_least": 0.15,
            "one_sided_permutation_p_at_most": 0.10,
            "top_minus_bottom_utility_positive": True,
        },
    }
    run_dir, _ = create_run_manifest(
        output_root=args.output_root.resolve(),
        repo_root=ROOT,
        stage="h1a_formal_tulu96",
        config=run_config,
        seed=args.seed,
        command=[sys.executable, *sys.argv],
        dataset_revisions=scoring_manifest["dataset_revisions"],
        model_revision=scoring_manifest["model_revision"],
        extra={
            "scoring_run_dir": str(scoring_dir),
            "utility_run_dir": str(utility_dir),
            "scoring_git_commit": scoring_manifest["git_commit"],
            "utility_git_commit": utility_manifest["git_commit"],
            "embeddings_sha256": _sha256_file(embeddings_path),
            "utility_measurements_sha256": _sha256_file(
                utility_dir / "utility_measurements.jsonl"
            ),
        },
    )
    metrics = {
        "candidate_count": 96,
        "query_count": len(all_queries),
        "error_query_count": len(error_indices),
        "observed_partial_spearman": observed_rho,
        "permutation_count": args.permutations,
        "permutation_greater_or_equal_count": greater_or_equal_count,
        "one_sided_permutation_p": permutation_p,
        "top_24_mean_utility": (
            sum(utilities[index] for index in top_indices) / 24
        ),
        "bottom_24_mean_utility": (
            sum(utilities[index] for index in bottom_indices) / 24
        ),
        "top_minus_bottom_mean_utility": top_bottom_difference,
        "rho_gate_passed": rho_gate,
        "permutation_gate_passed": permutation_gate,
        "direction_gate_passed": direction_gate,
        "h1a_gate_passed": h1a_passed,
        "claim_boundary": (
            "This is the preregistered Tulu-pool candidate-utility H1a test. "
            "It does not establish GSM8K-domain robustness or downstream SFT gains."
        ),
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_jsonl(run_dir / "candidate_analysis.jsonl", candidate_rows)
    _write_jsonl(
        run_dir / "permutation_statistics.jsonl",
        permutation_rows,
    )
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()
