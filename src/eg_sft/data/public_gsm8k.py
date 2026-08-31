"""Deterministic, text-free manifests for the public GSM8K protocol."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Normalize text for hashing and exact-leakage checks."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def validate_gsm8k_source_row(
    record: dict[str, Any],
    source_row: dict[str, str],
) -> None:
    """Verify that a pinned dataset row matches both frozen source hashes."""

    record_id = str(record.get("record_id", "unknown"))
    if sha256_text(source_row["question"]) != record.get("question_sha256"):
        raise ValueError(f"question hash mismatch for {record_id}")
    if sha256_text(source_row["answer"]) != record.get("answer_sha256"):
        raise ValueError(f"answer hash mismatch for {record_id}")


def word_ngrams(text: str, *, n: int = 5) -> set[tuple[str, ...]]:
    """Return normalized word n-grams, falling back to one short token tuple."""

    if n <= 0:
        raise ValueError("n must be positive")
    tokens = _WORD.findall(normalize_text(text))
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {
        tuple(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    }


def build_ngram_reference_index(
    texts: Sequence[str],
    *,
    n: int = 5,
) -> tuple[list[set[tuple[str, ...]]], dict[tuple[str, ...], set[int]]]:
    reference_sets = [word_ngrams(text, n=n) for text in texts]
    inverted: dict[tuple[str, ...], set[int]] = {}
    for reference_index, ngrams in enumerate(reference_sets):
        for ngram in ngrams:
            inverted.setdefault(ngram, set()).add(reference_index)
    return reference_sets, inverted


def maximum_ngram_overlap(
    text: str,
    *,
    reference_sets: Sequence[set[tuple[str, ...]]],
    inverted_index: dict[tuple[str, ...], set[int]],
    n: int = 5,
) -> tuple[int | None, float, float]:
    """Return best reference index, Jaccard, and smaller-set containment."""

    candidate = word_ngrams(text, n=n)
    if not candidate:
        return None, 0.0, 0.0

    possible_references: set[int] = set()
    for ngram in candidate:
        possible_references.update(inverted_index.get(ngram, ()))

    best_index: int | None = None
    best_jaccard = 0.0
    best_containment = 0.0
    for reference_index in possible_references:
        reference = reference_sets[reference_index]
        intersection = len(candidate & reference)
        if intersection == 0:
            continue
        union = len(candidate | reference)
        jaccard = intersection / union
        containment = intersection / min(len(candidate), len(reference))
        if (containment, jaccard, -reference_index) > (
            best_containment,
            best_jaccard,
            -(best_index if best_index is not None else reference_index),
        ):
            best_index = reference_index
            best_jaccard = jaccard
            best_containment = containment
    return best_index, best_jaccard, best_containment


def build_gsm8k_split_records(
    *,
    train_rows: Sequence[dict[str, str]],
    test_rows: Sequence[dict[str, str]],
    split_sizes: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign all train rows once and record test hashes without exposing text."""

    if sum(split_sizes.values()) != len(train_rows):
        raise ValueError(
            f"split sizes sum to {sum(split_sizes.values())}, "
            f"but train has {len(train_rows)} rows"
        )
    if any(size < 0 for size in split_sizes.values()):
        raise ValueError("split sizes must be non-negative")

    indices = list(range(len(train_rows)))
    random.Random(seed).shuffle(indices)

    split_by_index: dict[int, str] = {}
    cursor = 0
    for split_name, size in split_sizes.items():
        for index in indices[cursor : cursor + size]:
            split_by_index[index] = split_name
        cursor += size

    records: list[dict[str, Any]] = []
    for index, row in enumerate(train_rows):
        question_hash = sha256_text(row["question"])
        answer_hash = sha256_text(row["answer"])
        records.append(
            {
                "record_id": f"gsm8k-train-{index:04d}-{question_hash[:12]}",
                "source_split": "train",
                "source_index": index,
                "protocol_split": split_by_index[index],
                "question_sha256": question_hash,
                "answer_sha256": answer_hash,
            }
        )

    test_records: list[dict[str, Any]] = []
    for index, row in enumerate(test_rows):
        question_hash = sha256_text(row["question"])
        test_records.append(
            {
                "record_id": f"gsm8k-test-{index:04d}-{question_hash[:12]}",
                "source_split": "test",
                "source_index": index,
                "protocol_split": "held_out_test",
                "question_sha256": question_hash,
                "answer_sha256": sha256_text(row["answer"]),
            }
        )

    all_records = records + test_records
    if len({record["record_id"] for record in all_records}) != len(all_records):
        raise AssertionError("record IDs are not unique")

    manifest = {
        "seed": seed,
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "protocol_split_counts": {
            split_name: sum(
                record["protocol_split"] == split_name for record in records
            )
            for split_name in split_sizes
        },
        "held_out_test_count": len(test_records),
        "question_hash_overlap_train_test": len(
            {record["question_sha256"] for record in records}
            & {record["question_sha256"] for record in test_records}
        ),
    }
    return all_records, manifest


def _last_message(messages: Sequence[dict[str, str]], role: str) -> str | None:
    for message in reversed(messages):
        if message.get("role") == role and message.get("content", "").strip():
            return message["content"]
    return None


def candidate_prompt_text(messages: Sequence[dict[str, str]]) -> str:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("candidate must end with an assistant response")
    prompt_messages = messages[:-1]
    if not prompt_messages:
        raise ValueError("candidate has no prompt messages")
    return "\n".join(
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in prompt_messages
    )


def build_tulu_candidate_pool(
    *,
    rows: Iterable[dict[str, Any]],
    pool_size: int,
    seed: int,
    excluded_user_prompt_hashes: set[str],
    excluded_reference_texts: Sequence[str] = (),
    fuzzy_ngram_size: int = 5,
    fuzzy_threshold: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact-deduplicate Tulu rows and select a deterministic text-free pool."""

    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must be in [0, 1]")

    reference_sets, inverted_index = build_ngram_reference_index(
        excluded_reference_texts,
        n=fuzzy_ngram_size,
    )
    reference_hashes = [sha256_text(text) for text in excluded_reference_texts]

    unique_by_prompt: dict[str, dict[str, Any]] = {}
    invalid_rows = 0
    exact_gsm8k_exclusions = 0

    for fallback_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            invalid_rows += 1
            continue
        try:
            prompt = candidate_prompt_text(messages)
        except ValueError:
            invalid_rows += 1
            continue

        user_prompt = _last_message(messages[:-1], "user")
        response = _last_message(messages, "assistant")
        if user_prompt is None or response is None:
            invalid_rows += 1
            continue

        user_prompt_hash = sha256_text(user_prompt)
        if user_prompt_hash in excluded_user_prompt_hashes:
            exact_gsm8k_exclusions += 1
            continue

        prompt_hash = sha256_text(prompt)
        source_id = str(row.get("id") or f"row-{fallback_index}")
        source_index = int(row.get("index", fallback_index))
        candidate = {
            "candidate_id": f"tulu-{source_index:06d}-{prompt_hash[:12]}",
            "source_dataset": str(row.get("dataset", "unknown")),
            "source_id": source_id,
            "source_index": source_index,
            "prompt_sha256": prompt_hash,
            "user_prompt_sha256": user_prompt_hash,
            "response_sha256": sha256_text(response),
            "_user_prompt_text": user_prompt,
        }

        previous = unique_by_prompt.get(prompt_hash)
        if previous is None or (
            candidate["source_index"],
            candidate["source_id"],
        ) < (
            previous["source_index"],
            previous["source_id"],
        ):
            unique_by_prompt[prompt_hash] = candidate

    if len(unique_by_prompt) < pool_size:
        raise ValueError(
            f"only {len(unique_by_prompt)} unique candidates for pool size {pool_size}"
        )

    def priority(candidate: dict[str, Any]) -> str:
        material = (
            f"{seed}\0{candidate['source_id']}\0{candidate['prompt_sha256']}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    ranked = sorted(
        unique_by_prompt.values(),
        key=lambda candidate: (
            priority(candidate),
            candidate["source_index"],
            candidate["source_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    fuzzy_exclusions: list[dict[str, Any]] = []
    fuzzy_exclusion_count = 0
    for candidate in ranked:
        reference_index, jaccard, containment = maximum_ngram_overlap(
            candidate["_user_prompt_text"],
            reference_sets=reference_sets,
            inverted_index=inverted_index,
            n=fuzzy_ngram_size,
        )
        if reference_index is not None and (
            jaccard >= fuzzy_threshold or containment >= fuzzy_threshold
        ):
            fuzzy_exclusion_count += 1
            if len(fuzzy_exclusions) < 20:
                fuzzy_exclusions.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "user_prompt_sha256": candidate["user_prompt_sha256"],
                        "matched_reference_sha256": reference_hashes[reference_index],
                        "ngram_jaccard": jaccard,
                        "smaller_set_containment": containment,
                    }
                )
            continue

        candidate.pop("_user_prompt_text")
        candidate["selection_rank"] = len(selected)
        candidate["selection_priority_sha256"] = priority(candidate)
        selected.append(candidate)
        if len(selected) == pool_size:
            break

    if len(selected) < pool_size:
        raise ValueError(
            f"only {len(selected)} candidates remain after fuzzy leakage filtering"
        )

    manifest = {
        "seed": seed,
        "requested_pool_size": pool_size,
        "selected_pool_size": len(selected),
        "unique_prompt_count_before_sampling": len(unique_by_prompt),
        "invalid_row_count": invalid_rows,
        "exact_gsm8k_user_prompt_exclusions": exact_gsm8k_exclusions,
        "fuzzy_ngram_size": fuzzy_ngram_size,
        "fuzzy_threshold": fuzzy_threshold,
        "fuzzy_gsm8k_exclusion_count_before_pool_filled": fuzzy_exclusion_count,
        "fuzzy_exclusion_examples": fuzzy_exclusions,
        "selection_rule": (
            "ascending_sha256(seed + source_id + normalized_prompt_sha256)"
        ),
    }
    return selected, manifest


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
