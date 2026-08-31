"""Frozen B=500 selection validation and training-data construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eg_sft.data.public_gsm8k import candidate_prompt_text, sha256_text
from eg_sft.training.response_only import tokenize_response_only
from eg_sft.training.tulu import tulu_response_only_parts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(row)
    return rows


def validate_candidate_source(
    candidate: dict[str, Any],
    raw_row: dict[str, Any],
) -> list[dict[str, str]]:
    messages = raw_row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{candidate['candidate_id']} has invalid messages")
    if sha256_text(candidate_prompt_text(messages)) != candidate["prompt_sha256"]:
        raise ValueError(f"prompt hash mismatch for {candidate['candidate_id']}")
    response = str(messages[-1].get("content", ""))
    if sha256_text(response) != candidate["response_sha256"]:
        raise ValueError(f"response hash mismatch for {candidate['candidate_id']}")
    return messages


def tokenize_tulu_candidate(
    *,
    tokenizer: Any,
    candidate: dict[str, Any],
    raw_row: dict[str, Any],
    max_length: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    messages = validate_candidate_source(candidate, raw_row)
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer has no EOS token")
    prompt, response = tulu_response_only_parts(
        messages,
        eos_token=tokenizer.eos_token,
    )
    tokenized = tokenize_response_only(
        tokenizer,
        prompt=prompt,
        response=response,
        max_length=max_length,
        add_eos=True,
    )
    supervised_tokens = sum(label != -100 for label in tokenized["labels"])
    return tokenized, {
        "candidate_id": candidate["candidate_id"],
        "source_index": int(candidate["source_index"]),
        "prompt_sha256": candidate["prompt_sha256"],
        "response_sha256": candidate["response_sha256"],
        "total_tokens": len(tokenized["input_ids"]),
        "supervised_tokens": supervised_tokens,
    }


def validate_selection_manifest(
    manifest: dict[str, Any],
    *,
    expected_strategy: str,
    expected_budget: int,
    expected_selection_seed: int,
) -> list[dict[str, Any]]:
    if manifest.get("strategy") != expected_strategy:
        raise ValueError("selection strategy does not match requested strategy")
    if int(manifest.get("budget", -1)) != expected_budget:
        raise ValueError("selection budget does not match frozen budget")
    if int(manifest.get("selection_seed", -1)) != expected_selection_seed:
        raise ValueError("selection seed does not match frozen selection seed")
    selected = manifest.get("selected_candidates")
    if not isinstance(selected, list) or len(selected) != expected_budget:
        raise ValueError("selection manifest does not contain the frozen budget")
    candidate_ids = [str(row.get("candidate_id", "")) for row in selected]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("selected candidate IDs must be non-empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("selected candidate IDs must be unique")
    return selected


def selected_id_sha256(candidates: Sequence[dict[str, Any]]) -> str:
    payload = "\n".join(str(row["candidate_id"]) for row in candidates) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
