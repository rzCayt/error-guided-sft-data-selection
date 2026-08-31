from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.data.public_gsm8k import (  # noqa: E402
    build_gsm8k_split_records,
    build_tulu_candidate_pool,
    sha256_text,
    write_jsonl_exclusive,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic, text-free GSM8K/Tulu manifests."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-tulu",
        action="store_true",
        help="Prepare only GSM8K splits for a fast smoke test.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = _load_config(config_path)

    gsm_config = config["datasets"]["gsm8k"]
    gsm = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        revision=gsm_config["revision"],
    )
    train_rows = list(gsm["train"])
    test_rows = list(gsm["test"])
    gsm_records, gsm_manifest = build_gsm8k_split_records(
        train_rows=train_rows,
        test_rows=test_rows,
        split_sizes=config["gsm8k_train_splits"],
        seed=int(config["seed"]),
    )
    write_jsonl_exclusive(output_dir / "gsm8k_records.jsonl", gsm_records)

    manifest: dict[str, Any] = {
        "protocol_version": config["protocol_version"],
        "config_path": str(config_path),
        "config": config,
        "gsm8k": gsm_manifest,
        "claim_boundary": (
            "Output files contain source IDs and hashes, not redistributed source text."
        ),
    }

    if not args.skip_tulu:
        tulu_config = config["datasets"]["candidate_pool"]
        tulu = load_dataset(
            tulu_config["repo_id"],
            tulu_config["config"],
            split="train",
            revision=tulu_config["revision"],
        )
        excluded_hashes = {
            sha256_text(row["question"]) for row in train_rows + test_rows
        }
        candidate_records, candidate_manifest = build_tulu_candidate_pool(
            rows=tulu,
            pool_size=int(config["candidate_pool_size"]),
            seed=int(config["seed"]),
            excluded_user_prompt_hashes=excluded_hashes,
            excluded_reference_texts=[
                row["question"] for row in train_rows + test_rows
            ],
            fuzzy_ngram_size=5,
            fuzzy_threshold=0.8,
        )
        write_jsonl_exclusive(
            output_dir / "tulu_candidate_pool.jsonl", candidate_records
        )
        manifest["candidate_pool"] = candidate_manifest

    with (output_dir / "data_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
