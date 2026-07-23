"""Compute scaled RDS+ scores for frozen all-query and error-query groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import sha256_text  # noqa: E402
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.selection.h1a_sample import stratified_candidate_sample  # noqa: E402
from eg_sft.selection.query_groups import load_jsonl  # noqa: E402
from eg_sft.selection.rds import (  # noqa: E402
    RDS_FORMAT_VERSION,
    cosine_similarity_matrix,
    encode_rds_texts,
    format_gsm8k_rds_text,
    format_tulu_rds_text,
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


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _candidate_prompt_for_hash(messages: Sequence[dict[str, str]]) -> str:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("candidate must end with assistant")
    prompt_messages = messages[:-1]
    if not prompt_messages:
        raise ValueError("candidate has no prompt messages")
    return "\n".join(
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in prompt_messages
    )


def _validate_and_format_candidate(
    *,
    candidate: dict[str, Any],
    raw_row: dict[str, Any],
    eos_token: str,
) -> str:
    messages = raw_row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{candidate['candidate_id']} has invalid messages")
    prompt_hash = sha256_text(_candidate_prompt_for_hash(messages))
    response_hash = sha256_text(str(messages[-1].get("content", "")))
    if prompt_hash != candidate["prompt_sha256"]:
        raise ValueError(f"prompt hash mismatch for {candidate['candidate_id']}")
    if response_hash != candidate["response_sha256"]:
        raise ValueError(f"response hash mismatch for {candidate['candidate_id']}")
    return format_tulu_rds_text(messages, eos_token=eos_token)


def _spearman_from_complete_ranks(left: Sequence[int], right: Sequence[int]) -> float:
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


def _top_jaccard(left: Sequence[int], right: Sequence[int], count: int) -> float:
    left_set = set(left[:count])
    right_set = set(right[:count])
    return len(left_set & right_set) / len(left_set | right_set)


def _reliability_indices(error_order: Sequence[int], count: int = 10) -> list[int]:
    if count <= 0 or count > len(error_order):
        raise ValueError("invalid reliability count")
    if count == 1:
        return [error_order[0]]
    positions = [
        round(index * (len(error_order) - 1) / (count - 1))
        for index in range(count)
    ]
    return [error_order[position] for position in positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--query-groups-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("RDS scoring requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("RDS scoring requires BF16 support")
    device = torch.device("cuda")

    config = _read_json(args.config.resolve())
    model_config = config["model"]
    gsm_config = config["datasets"]["gsm8k"]
    candidate_config = config["datasets"]["candidate_pool"]
    run_config = {
        "representation": {
            "version": RDS_FORMAT_VERSION,
            "model": model_config,
            "pooling": "last_hidden_state_weighted_mean",
            "padding_tokens_excluded": True,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        },
        "query_groups": ["all_query", "error_query"],
        "candidate_count": args.candidate_count,
        "candidate_sampling": "source_stratified_stable_hash_round_robin",
        "candidate_config": candidate_config,
        "gsm8k_config": gsm_config,
    }
    run_dir, _ = create_run_manifest(
        output_root=args.output_root.resolve(),
        repo_root=ROOT,
        stage="rds_h1a_scoring",
        config=run_config,
        seed=args.seed,
        command=[sys.executable, *sys.argv],
        dataset_revisions={
            gsm_config["repo_id"]: gsm_config["revision"],
            candidate_config["repo_id"]: candidate_config["revision"],
        },
        model_revision=model_config["revision"],
        extra={
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        },
    )

    set_seed(args.seed)
    candidate_pool = load_jsonl(
        args.data_manifest_dir.resolve() / "tulu_candidate_pool.jsonl"
    )
    candidates = stratified_candidate_sample(
        candidate_pool,
        count=args.candidate_count,
        seed=args.seed,
    )
    all_queries = load_jsonl(
        args.query_groups_dir.resolve() / "all_queries.jsonl"
    )
    error_queries = load_jsonl(
        args.query_groups_dir.resolve() / "error_queries.jsonl"
    )
    error_ids = {row["record_id"] for row in error_queries}
    if error_ids != {
        row["record_id"] for row in all_queries if not row["numeric_correct"]
    }:
        raise ValueError("error query list does not match all-query labels")

    gsm = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        split="train",
        revision=gsm_config["revision"],
    )
    tulu = load_dataset(
        candidate_config["repo_id"],
        candidate_config["config"],
        split="train",
        revision=candidate_config["revision"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer has no EOS token")

    query_texts: list[str] = []
    for query in all_queries:
        row = gsm[int(query["source_index"])]
        if sha256_text(row["question"]) != query["question_sha256"]:
            raise ValueError(f"query hash mismatch for {query['record_id']}")
        query_texts.append(
            format_gsm8k_rds_text(
                question=row["question"],
                answer=row["answer"],
                eos_token=tokenizer.eos_token,
            )
        )
    candidate_texts = [
        _validate_and_format_candidate(
            candidate=candidate,
            raw_row=tulu[int(candidate["source_index"])],
            eos_token=tokenizer.eos_token,
        )
        for candidate in candidates
    ]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    started = time.perf_counter()
    query_embeddings = encode_rds_texts(
        model=model,
        tokenizer=tokenizer,
        texts=query_texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    candidate_embeddings = encode_rds_texts(
        model=model,
        tokenizer=tokenizer,
        texts=candidate_texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated())

    error_mask = torch.tensor(
        [query["record_id"] in error_ids for query in all_queries],
        dtype=torch.bool,
    )
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
    all_scores = rank_scores(all_order, candidate_count=len(candidates))
    error_scores = rank_scores(error_order, candidate_count=len(candidates))
    all_ranks = {candidate: rank for rank, candidate in enumerate(all_order)}
    error_ranks = {candidate: rank for rank, candidate in enumerate(error_order)}

    score_rows = [
        {
            "candidate_id": candidate["candidate_id"],
            "source_dataset": candidate["source_dataset"],
            "source_index": candidate["source_index"],
            "prompt_sha256": candidate["prompt_sha256"],
            "all_query_rank": all_ranks[index],
            "all_query_score": all_scores[index],
            "error_query_rank": error_ranks[index],
            "error_query_score": error_scores[index],
        }
        for index, candidate in enumerate(candidates)
    ]
    reliability_rows = [
        score_rows[index] for index in _reliability_indices(error_order, count=10)
    ]
    top_count = max(1, len(candidates) // 4)
    metrics = {
        "representation_version": RDS_FORMAT_VERSION,
        "candidate_count": len(candidates),
        "all_query_count": len(all_queries),
        "error_query_count": len(error_queries),
        "embedding_dimension": candidate_embeddings.shape[1],
        "all_vs_error_order_identical": all_order == error_order,
        "all_vs_error_rank_spearman": _spearman_from_complete_ranks(
            all_order,
            error_order,
        ),
        "all_vs_error_top_quartile_jaccard": _top_jaccard(
            all_order,
            error_order,
            top_count,
        ),
        "top_quartile_count": top_count,
        "query_embeddings_sha256": _tensor_sha256(query_embeddings),
        "candidate_embeddings_sha256": _tensor_sha256(candidate_embeddings),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_memory,
        "peak_memory_gib": peak_memory / 1024**3,
        "claim_boundary": (
            "Scores are a 96-candidate scaled RDS+ diagnostic. "
            "They do not establish candidate utility or selector effectiveness."
        ),
    }
    torch.save(
        {
            "query_embeddings": query_embeddings,
            "candidate_embeddings": candidate_embeddings,
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "query_ids": [row["record_id"] for row in all_queries],
        },
        run_dir / "embeddings.pt",
    )
    _write_jsonl(run_dir / "candidate_scores.jsonl", score_rows)
    _write_json(run_dir / "reliability_candidates.json", reliability_rows)
    _write_json(run_dir / "metrics.json", metrics)
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()
