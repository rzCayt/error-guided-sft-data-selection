"""Export v3 similarity and optionally build fuzzy prompt clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import candidate_prompt_text, sha256_text  # noqa: E402
from eg_sft.experiment.budget_equivalent_inputs import (  # noqa: E402
    cluster_near_duplicate_prompts,
    eligible_candidate_rows,
    export_similarity_artifact,
    write_jsonl_exclusive,
)
from eg_sft.experiment.budget_equivalent_protocol import read_json_object  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rds-run-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-config", type=Path, default=Path("configs/public_gsm8k_v1.json")
    )
    parser.add_argument(
        "--candidate-inventory",
        type=Path,
        default=Path(
            "results/research_public_gsm8k_v1/rds_full_pool_10k_public_evidence_v1/"
            "artifacts/candidate_inventory.jsonl"
        ),
    )
    parser.add_argument(
        "--query-inventory",
        type=Path,
        default=Path(
            "results/research_public_gsm8k_v1/rds_full_pool_10k_public_evidence_v1/"
            "artifacts/query_inventory.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".aris/compute/budget_equivalent_v3_inputs"),
    )
    parser.add_argument("--build-near-duplicate-clusters", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    similarity_path = output_dir / "query_candidate_similarity.pt"
    similarity = export_similarity_artifact(
        run_dir=args.rds_run_dir.resolve(),
        query_inventory_path=args.query_inventory.resolve(),
        candidate_inventory_path=args.candidate_inventory.resolve(),
        output_path=similarity_path,
    )
    result: dict[str, object] = {"similarity_artifact": similarity}
    if args.build_near_duplicate_clusters:
        protocol = read_json_object(args.protocol_config.resolve())
        source = protocol["datasets"]["candidate_pool"]
        dataset = load_dataset(
            source["repo_id"],
            source["config"],
            split="train",
            revision=source["revision"],
        )
        candidates = eligible_candidate_rows(
            read_jsonl(args.candidate_inventory.resolve())
        )
        candidate_ids, prompts = [], []
        for candidate in candidates:
            raw = dataset[int(candidate["source_index"])]
            messages = raw.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"invalid source messages for {candidate['candidate_id']}")
            prompt = candidate_prompt_text(messages)
            if sha256_text(prompt) != candidate["user_prompt_sha256"]:
                raise ValueError(f"source prompt hash changed for {candidate['candidate_id']}")
            candidate_ids.append(str(candidate["candidate_id"]))
            prompts.append(prompt)
        clusters, cluster_audit = cluster_near_duplicate_prompts(candidate_ids, prompts)
        cluster_path = output_dir / "near_duplicate_clusters.jsonl"
        write_jsonl_exclusive(cluster_path, clusters)
        cluster_audit["path"] = str(cluster_path)
        cluster_audit["sha256"] = file_sha256(cluster_path)
        _write_json_exclusive(output_dir / "near_duplicate_cluster_audit.json", cluster_audit)
        result["near_duplicate_clusters"] = cluster_audit
    _write_json_exclusive(output_dir / "input_bindings.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
