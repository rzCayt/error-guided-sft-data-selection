"""Freeze the response-trainable random B=500 Tulu selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.selection.h1a_sample import (  # noqa: E402
    select_until_eligible_count,
    stratified_candidate_sample,
)
from eg_sft.training.b500 import (  # noqa: E402
    read_jsonl,
    selected_id_sha256,
    tokenize_tulu_candidate,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--recipe-config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = _read_json(args.protocol_config.resolve())
    recipe = _read_json(args.recipe_config.resolve())
    selection = recipe["selection"]
    training = recipe["training"]
    if recipe["engineering_closure"]["strategy"] != "random":
        raise ValueError("engineering closure must remain random")

    candidate_config = protocol["datasets"]["candidate_pool"]
    model_config = protocol["model"]
    pool = read_jsonl(
        args.data_manifest_dir.resolve() / "tulu_candidate_pool.jsonl"
    )
    if len(pool) != selection["candidate_pool_size"]:
        raise ValueError("candidate pool size does not match frozen recipe")
    frozen_order = stratified_candidate_sample(
        pool,
        count=len(pool),
        seed=int(selection["selection_seed"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tulu = load_dataset(
        candidate_config["repo_id"],
        candidate_config["config"],
        split="train",
        revision=candidate_config["revision"],
    )
    audits: dict[str, dict[str, Any]] = {}

    def is_trainable(candidate: dict[str, Any]) -> bool:
        try:
            _, audit = tokenize_tulu_candidate(
                tokenizer=tokenizer,
                candidate=candidate,
                raw_row=tulu[int(candidate["source_index"])],
                max_length=int(training["max_length"]),
            )
        except ValueError as error:
            if "response was fully truncated" not in str(error):
                raise
            audit = {
                "candidate_id": candidate["candidate_id"],
                "source_index": int(candidate["source_index"]),
                "prompt_sha256": candidate["prompt_sha256"],
                "response_sha256": candidate["response_sha256"],
                "total_tokens": int(training["max_length"]),
                "supervised_tokens": 0,
            }
        audits[candidate["candidate_id"]] = audit
        return int(audit["supervised_tokens"]) > 0

    selected, excluded = select_until_eligible_count(
        frozen_order,
        count=int(selection["budget"]),
        is_eligible=is_trainable,
    )
    selected_rows = [
        {
            **candidate,
            "total_tokens": audits[candidate["candidate_id"]]["total_tokens"],
            "supervised_tokens": audits[candidate["candidate_id"]][
                "supervised_tokens"
            ],
        }
        for candidate in selected
    ]
    payload = {
        "protocol_version": recipe["protocol_version"],
        "strategy": "random",
        "budget": int(selection["budget"]),
        "selection_seed": int(selection["selection_seed"]),
        "selection_rule": selection["random_rule"],
        "candidate_scan_count": len(selected) + len(excluded),
        "excluded_fully_truncated_count": len(excluded),
        "excluded_candidate_ids": [
            candidate["candidate_id"] for candidate in excluded
        ],
        "selected_id_sha256": selected_id_sha256(selected_rows),
        "selected_candidates": selected_rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "selected": len(selected_rows),
                "excluded": len(excluded),
                "selected_id_sha256": payload["selected_id_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
