"""Freeze text-free arithmetic OOD manifests and contamination audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import (  # noqa: E402
    build_ngram_reference_index,
    maximum_ngram_overlap,
    normalize_text,
    sha256_text,
)
from eg_sft.evaluation.arithmetic_ood import (  # noqa: E402
    build_ood_record,
    source_question,
)
from eg_sft.experiment.budget_equivalent_inputs import (  # noqa: E402
    cluster_near_duplicate_prompts,
    write_jsonl_exclusive,
)
from eg_sft.experiment.budget_equivalent_protocol import read_json_object  # noqa: E402
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("Tulu source messages are invalid")
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content", "")).strip():
            return str(message["content"])
    raise ValueError("Tulu source row has no user message")


def _canonical_row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood-config", type=Path, required=True)
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
        "--gsm8k-records",
        type=Path,
        default=Path(
            "results/research_public_gsm8k_v1/data_manifest_full_v2_fuzzy/"
            "gsm8k_records.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ood_config_path = args.ood_config.resolve()
    protocol_path = args.protocol_config.resolve()
    candidate_inventory_path = args.candidate_inventory.resolve()
    gsm_records_path = args.gsm8k_records.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    ood_config = read_json_object(ood_config_path)
    protocol = read_json_object(protocol_path)

    candidate_inventory = [
        row
        for row in read_jsonl(candidate_inventory_path)
        if row.get("response_only_trainable") is True
    ]
    tulu_spec = protocol["datasets"]["candidate_pool"]
    tulu = load_dataset(
        tulu_spec["repo_id"],
        tulu_spec["config"],
        split="train",
        revision=tulu_spec["revision"],
    )
    reference_texts = []
    for candidate in candidate_inventory:
        text = _last_user_text(tulu[int(candidate["source_index"])]["messages"])
        if sha256_text(text) != candidate["user_prompt_sha256"]:
            raise ValueError(f"candidate prompt hash changed: {candidate['candidate_id']}")
        reference_texts.append(text)

    gsm_spec = protocol["datasets"]["gsm8k"]
    gsm_splits = {
        split: load_dataset(
            gsm_spec["repo_id"],
            gsm_spec["config"],
            split=split,
            revision=gsm_spec["revision"],
        )
        for split in ("train", "test")
    }
    for record in read_jsonl(gsm_records_path):
        source_split = str(record["source_split"])
        question = str(gsm_splits[source_split][int(record["source_index"])]["question"])
        if sha256_text(question) != record["question_sha256"]:
            raise ValueError(f"GSM8K question hash changed: {record['record_id']}")
        reference_texts.append(question)

    decontamination = ood_config["decontamination"]
    ngram_size = int(decontamination["ngram_size"])
    reference_sets, inverted_index = build_ngram_reference_index(
        reference_texts, n=ngram_size
    )
    reference_hashes = [sha256_text(normalize_text(text)) for text in reference_texts]
    exact_reference_hashes = set(reference_hashes)

    eligible_rows: list[dict[str, Any]] = []
    excluded_non_numeric: list[dict[str, Any]] = []
    excluded_reference_overlap: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    dataset_order = list(ood_config["datasets"])
    for dataset_name in dataset_order:
        spec = ood_config["datasets"][dataset_name]
        source = load_dataset(
            spec["repo_id"],
            spec["config"],
            split=spec["split"],
            revision=spec["revision"],
        )
        if len(source) != int(spec["source_count"]):
            raise ValueError(f"{dataset_name} source count changed")
        numeric_count = 0
        overlap_count = 0
        for source_index, raw_row in enumerate(source):
            raw = dict(raw_row)
            record = build_ood_record(
                dataset=dataset_name,
                source_index=source_index,
                row=raw,
                answer_field=str(spec["answer_field"]),
            )
            record["source_row_sha256"] = _canonical_row_sha256(raw)
            if not record["numeric_eligible"]:
                excluded_non_numeric.append(record)
                continue
            numeric_count += 1
            question = source_question(dataset_name, raw)
            normalized_hash = sha256_text(normalize_text(question))
            match_index, jaccard, containment = maximum_ngram_overlap(
                question,
                reference_sets=reference_sets,
                inverted_index=inverted_index,
                n=ngram_size,
            )
            is_exact = normalized_hash in exact_reference_hashes
            is_fuzzy = match_index is not None and (
                jaccard >= float(decontamination["jaccard_threshold"])
                or containment >= float(decontamination["containment_threshold"])
            )
            if is_exact or is_fuzzy:
                overlap_count += 1
                excluded_reference_overlap.append(
                    {
                        "record_id": record["record_id"],
                        "dataset": dataset_name,
                        "question_sha256": record["question_sha256"],
                        "matched_reference_normalized_sha256": (
                            reference_hashes[match_index] if match_index is not None else normalized_hash
                        ),
                        "exact_normalized_match": is_exact,
                        "ngram_jaccard": jaccard,
                        "smaller_set_containment": containment,
                    }
                )
                continue
            record["gold_value_sha256"] = sha256_text(str(record.pop("gold_value")))
            record.pop("numeric_eligible", None)
            eligible_rows.append(record | {"_question_text": question})
        expected_numeric = spec.get("expected_unique_numeric_count_before_decontamination")
        if expected_numeric is not None and numeric_count != int(expected_numeric):
            raise ValueError(f"{dataset_name} numeric-eligible count changed")
        if expected_numeric is None and numeric_count != len(source):
            raise ValueError(f"{dataset_name} unexpectedly contains non-numeric gold")
        source_audit[dataset_name] = {
            "source_count": len(source),
            "numeric_eligible_count": numeric_count,
            "reference_overlap_exclusion_count": overlap_count,
        }

    cluster_rows, cluster_audit = cluster_near_duplicate_prompts(
        [str(row["record_id"]) for row in eligible_rows],
        [str(row["_question_text"]) for row in eligible_rows],
        ngram_size=ngram_size,
        jaccard_threshold=float(decontamination["jaccard_threshold"]),
        containment_threshold=float(decontamination["containment_threshold"]),
    )
    cluster_by_id = {
        row["candidate_id"]: row["near_duplicate_cluster_id"] for row in cluster_rows
    }
    retained = []
    excluded_cross_ood = []
    seen_clusters: set[str] = set()
    for row in eligible_rows:
        cluster_id = cluster_by_id[str(row["record_id"])]
        public_row = {key: value for key, value in row.items() if not key.startswith("_")}
        public_row["near_duplicate_cluster_id"] = cluster_id
        if cluster_id in seen_clusters:
            excluded_cross_ood.append(public_row)
            continue
        seen_clusters.add(cluster_id)
        retained.append(public_row)

    counts = Counter(str(row["dataset"]) for row in retained)
    artifact_bindings = {}
    for dataset_name in dataset_order:
        path = output_dir / f"{dataset_name}_records.jsonl"
        rows = [row for row in retained if row["dataset"] == dataset_name]
        write_jsonl_exclusive(path, rows)
        artifact_bindings[dataset_name] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "retained_count": len(rows),
        }
    write_jsonl_exclusive(output_dir / "excluded_non_numeric.jsonl", excluded_non_numeric)
    write_jsonl_exclusive(
        output_dir / "excluded_reference_overlap.jsonl", excluded_reference_overlap
    )
    write_jsonl_exclusive(output_dir / "excluded_cross_ood_duplicate.jsonl", excluded_cross_ood)
    manifest = {
        "schema_version": "budget-equivalent-arithmetic-ood-manifest-v1",
        "ood_config_sha256": file_sha256(ood_config_path),
        "protocol_config_sha256": file_sha256(protocol_path),
        "candidate_inventory_sha256": file_sha256(candidate_inventory_path),
        "gsm8k_records_sha256": file_sha256(gsm_records_path),
        "reference_text_count": len(reference_texts),
        "source_audit": source_audit,
        "retained_counts": dict(sorted(counts.items())),
        "retained_total": len(retained),
        "excluded_non_numeric_count": len(excluded_non_numeric),
        "excluded_reference_overlap_count": len(excluded_reference_overlap),
        "excluded_cross_ood_duplicate_count": len(excluded_cross_ood),
        "cross_ood_cluster_audit": cluster_audit,
        "artifacts": artifact_bindings,
        "raw_dataset_text_stored": False,
        "claim_boundary": (
            "These text-free manifests freeze arithmetic OOD membership and "
            "mechanical decontamination only; they contain no model results."
        ),
    }
    _write_json_exclusive(output_dir / "ood_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
