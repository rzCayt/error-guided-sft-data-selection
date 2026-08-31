"""Freeze and score 48 GSM8K in-domain candidates for the H1a boundary check."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import (  # noqa: E402
    build_ngram_reference_index,
    maximum_ngram_overlap,
    validate_gsm8k_source_row,
)
from eg_sft.experiment.run_manifest import create_run_manifest  # noqa: E402
from eg_sft.selection.h1a_sample import (  # noqa: E402
    select_until_eligible_count,
    stable_record_order,
)
from eg_sft.selection.query_groups import load_jsonl  # noqa: E402
from eg_sft.selection.rds import (  # noqa: E402
    RDS_FORMAT_VERSION,
    cosine_similarity_matrix,
    encode_rds_texts,
    format_gsm8k_rds_text,
    rank_scores,
    round_robin_order,
)
from eg_sft.training.overfit import gsm8k_training_text  # noqa: E402
from eg_sft.training.response_only import tokenize_response_only  # noqa: E402


DOMAIN_NAMESPACE = "gsm8k-domain-h1a-v1"
FUZZY_NGRAM_SIZE = 5
FUZZY_THRESHOLD = 0.8


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


def _rank_spearman(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or set(left) != set(right):
        raise ValueError("orders must contain the same candidates")
    count = len(left)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--query-groups-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    if args.candidate_count != 48:
        raise ValueError("the frozen GSM8K domain boundary requires 48 candidates")
    if not torch.cuda.is_available():
        raise RuntimeError("RDS scoring requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("RDS scoring requires BF16")
    device = torch.device("cuda")

    config = _read_json(args.config.resolve())
    model_config = config["model"]
    gsm_config = config["datasets"]["gsm8k"]
    run_config = {
        "scope": "gsm8k_in_domain_candidate_pool",
        "candidate_count": 48,
        "candidate_order": (
            "ascending_sha256(seed + namespace + record_id)"
        ),
        "candidate_namespace": DOMAIN_NAMESPACE,
        "eligibility": {
            "response_only_trainable_at_max_length": args.max_length,
            "exclude_exact_question_overlap_with_diagnostic_or_utility": True,
            "fuzzy_ngram_size": FUZZY_NGRAM_SIZE,
            "fuzzy_jaccard_or_containment_threshold": FUZZY_THRESHOLD,
        },
        "representation": {
            "version": RDS_FORMAT_VERSION,
            "model": model_config,
            "pooling": "last_hidden_state_weighted_mean",
            "padding_tokens_excluded": True,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        },
        "query_groups": ["all_query", "error_query"],
        "gsm8k_config": gsm_config,
    }
    run_dir, manifest = create_run_manifest(
        output_root=args.output_root.resolve(),
        repo_root=ROOT,
        stage="rds_h1a_gsm8k_domain48_scoring",
        config=run_config,
        seed=args.seed,
        command=[sys.executable, *sys.argv],
        dataset_revisions={gsm_config["repo_id"]: gsm_config["revision"]},
        model_revision=model_config["revision"],
        extra={
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        },
    )

    try:
        records = load_jsonl(
            args.data_manifest_dir.resolve() / "gsm8k_records.jsonl"
        )
        record_by_id = {row["record_id"]: row for row in records}
        if len(record_by_id) != len(records):
            raise ValueError("GSM8K record IDs are not unique")
        domain_pool = [
            row
            for row in records
            if row["protocol_split"] == "in_domain_candidate_pool"
        ]
        utility_records = [
            row
            for row in records
            if row["protocol_split"] == "candidate_utility_validation"
        ]
        if len(domain_pool) != 6705 or len(utility_records) != 128:
            raise ValueError(
                "frozen GSM8K split counts differ from 6705 domain / 128 utility"
            )
        frozen_order = stable_record_order(
            domain_pool,
            id_field="record_id",
            seed=args.seed,
            namespace=DOMAIN_NAMESPACE,
        )
        all_queries = load_jsonl(
            args.query_groups_dir.resolve() / "all_queries.jsonl"
        )
        error_queries = load_jsonl(
            args.query_groups_dir.resolve() / "error_queries.jsonl"
        )
        if len(all_queries) != 448 or len(error_queries) != 99:
            raise ValueError("frozen query groups must contain 448 all / 99 error")
        error_ids = {row["record_id"] for row in error_queries}
        if error_ids != {
            row["record_id"] for row in all_queries if not row["numeric_correct"]
        }:
            raise ValueError("error queries differ from frozen numeric labels")

        gsm = load_dataset(
            gsm_config["repo_id"],
            gsm_config["config"],
            split="train",
            revision=gsm_config["revision"],
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

        query_rows: list[dict[str, str]] = []
        query_texts: list[str] = []
        for query in all_queries:
            record = record_by_id[query["record_id"]]
            if record["protocol_split"] != "selection_diagnostic":
                raise ValueError(f"{query['record_id']} is not diagnostic")
            row = gsm[int(record["source_index"])]
            validate_gsm8k_source_row(record, row)
            query_rows.append(row)
            query_texts.append(
                format_gsm8k_rds_text(
                    question=row["question"],
                    answer=row["answer"],
                    eos_token=tokenizer.eos_token,
                )
            )

        utility_rows = [
            gsm[int(record["source_index"])] for record in utility_records
        ]
        for record, row in zip(utility_records, utility_rows, strict=True):
            validate_gsm8k_source_row(record, row)
        reference_records = [
            record_by_id[query["record_id"]] for query in all_queries
        ] + utility_records
        reference_rows = query_rows + utility_rows
        reference_hashes = {
            record["question_sha256"] for record in reference_records
        }
        reference_ngram_sets, reference_inverted_index = (
            build_ngram_reference_index(
                [row["question"] for row in reference_rows],
                n=FUZZY_NGRAM_SIZE,
            )
        )

        candidate_cache: dict[str, dict[str, Any]] = {}

        def candidate_is_eligible(record: dict[str, Any]) -> bool:
            row = gsm[int(record["source_index"])]
            validate_gsm8k_source_row(record, row)
            audit: dict[str, Any] = {
                "candidate_id": record["record_id"],
                "source_index": record["source_index"],
                "response_only_trainable": False,
                "training_total_tokens": args.max_length,
                "training_supervised_tokens": 0,
                "reason": None,
            }
            if record["question_sha256"] in reference_hashes:
                audit["reason"] = "exact_question_overlap"
                candidate_cache[record["record_id"]] = audit
                return False
            matched_index, jaccard, containment = maximum_ngram_overlap(
                row["question"],
                reference_sets=reference_ngram_sets,
                inverted_index=reference_inverted_index,
                n=FUZZY_NGRAM_SIZE,
            )
            audit["maximum_ngram_jaccard"] = jaccard
            audit["maximum_smaller_set_containment"] = containment
            audit["matched_reference_id"] = (
                reference_records[matched_index]["record_id"]
                if matched_index is not None
                else None
            )
            if (
                jaccard >= FUZZY_THRESHOLD
                or containment >= FUZZY_THRESHOLD
            ):
                audit["reason"] = "fuzzy_question_overlap"
                candidate_cache[record["record_id"]] = audit
                return False

            prompt, response = gsm8k_training_text(
                row["question"],
                row["answer"],
            )
            try:
                tokenized = tokenize_response_only(
                    tokenizer,
                    prompt=prompt,
                    response=response,
                    max_length=args.max_length,
                    add_eos=True,
                )
            except ValueError as error:
                if "response was fully truncated" not in str(error):
                    raise
                audit["reason"] = "response_fully_truncated"
                candidate_cache[record["record_id"]] = audit
                return False
            audit.update(
                {
                    "response_only_trainable": True,
                    "training_total_tokens": len(tokenized["input_ids"]),
                    "training_supervised_tokens": sum(
                        label != -100 for label in tokenized["labels"]
                    ),
                    "reason": "selected",
                    "rds_text": format_gsm8k_rds_text(
                        question=row["question"],
                        answer=row["answer"],
                        eos_token=tokenizer.eos_token,
                    ),
                }
            )
            candidate_cache[record["record_id"]] = audit
            return True

        candidates, excluded = select_until_eligible_count(
            frozen_order,
            count=48,
            is_eligible=candidate_is_eligible,
        )
        candidate_texts = [
            candidate_cache[row["record_id"]]["rds_text"] for row in candidates
        ]

        set_seed(args.seed)
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
        all_scores = rank_scores(all_order, candidate_count=48)
        error_scores = rank_scores(error_order, candidate_count=48)
        all_ranks = {candidate: rank for rank, candidate in enumerate(all_order)}
        error_ranks = {
            candidate: rank for rank, candidate in enumerate(error_order)
        }
        score_rows = [
            {
                "candidate_id": record["record_id"],
                "source_dataset": "openai/gsm8k",
                "source_index": record["source_index"],
                "question_sha256": record["question_sha256"],
                "answer_sha256": record["answer_sha256"],
                "all_query_rank": all_ranks[index],
                "all_query_score": all_scores[index],
                "error_query_rank": error_ranks[index],
                "error_query_score": error_scores[index],
                "response_only_trainable": True,
                "training_total_tokens": candidate_cache[
                    record["record_id"]
                ]["training_total_tokens"],
                "training_supervised_tokens": candidate_cache[
                    record["record_id"]
                ]["training_supervised_tokens"],
                "maximum_ngram_jaccard": candidate_cache[
                    record["record_id"]
                ]["maximum_ngram_jaccard"],
                "maximum_smaller_set_containment": candidate_cache[
                    record["record_id"]
                ]["maximum_smaller_set_containment"],
            }
            for index, record in enumerate(candidates)
        ]
        exclusion_rows = [
            {
                key: value
                for key, value in candidate_cache[record["record_id"]].items()
                if key != "rds_text"
            }
            for record in excluded
        ]
        reason_counts = Counter(row["reason"] for row in exclusion_rows)
        metrics = {
            "scope": "gsm8k_in_domain_candidate_pool",
            "candidate_count": 48,
            "candidate_scan_count": len(candidates) + len(excluded),
            "excluded_before_filling_count": len(excluded),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "all_query_count": len(all_queries),
            "error_query_count": len(error_queries),
            "embedding_dimension": candidate_embeddings.shape[1],
            "all_vs_error_order_identical": all_order == error_order,
            "all_vs_error_rank_spearman": _rank_spearman(
                all_order,
                error_order,
            ),
            "top_quartile_count": 12,
            "top_quartile_jaccard": _top_jaccard(
                all_order,
                error_order,
                12,
            ),
            "query_embeddings_sha256": _tensor_sha256(query_embeddings),
            "candidate_embeddings_sha256": _tensor_sha256(
                candidate_embeddings
            ),
            "selected_question_hash_overlap_with_references": len(
                {row["question_sha256"] for row in candidates}
                & reference_hashes
            ),
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_memory,
            "peak_memory_gib": peak_memory / 1024**3,
            "claim_boundary": (
                "This run freezes 48 GSM8K in-domain candidate scores. "
                "It does not measure utility or establish H1a."
            ),
        }
        torch.save(
            {
                "query_embeddings": query_embeddings,
                "candidate_embeddings": candidate_embeddings,
                "candidate_ids": [row["record_id"] for row in candidates],
                "query_ids": [row["record_id"] for row in all_queries],
            },
            run_dir / "embeddings.pt",
        )
        _write_jsonl(run_dir / "candidate_scores.jsonl", score_rows)
        _write_jsonl(run_dir / "candidate_preflight_exclusions.jsonl", exclusion_rows)
        _write_json(run_dir / "metrics.json", metrics)
        print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))
    except Exception as error:
        _write_json(
            run_dir / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "manifest": manifest,
            },
        )
        raise


if __name__ == "__main__":
    main()
