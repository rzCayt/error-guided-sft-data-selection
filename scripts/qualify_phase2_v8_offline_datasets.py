"""Verify every formal evaluation dataset is available and correct in offline mode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    read_json,
    write_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract  # noqa: E402
from eg_sft.experiment.budget_equivalent_ood_runtime import (  # noqa: E402
    OOD_DATASETS,
    resolve_ood_contract,
    validate_source_row,
)
from eg_sft.experiment.phase2_v7_environment import canonical_json_sha256  # noqa: E402
from eg_sft.training.b500 import read_jsonl  # noqa: E402


def _gsm8k(config_path: Path) -> dict:
    contract = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=config_path,
        cell_id="v8_rep1_random_common_mix_train17",
    )
    spec = contract["protocol"]["datasets"]["gsm8k"]
    train = load_dataset(
        spec["repo_id"],
        spec["config"],
        split="train",
        revision=spec["revision"],
    )
    test = load_dataset(
        spec["repo_id"],
        spec["config"],
        split="test",
        revision=spec["revision"],
    )
    records = read_jsonl(contract["data_dir"] / "gsm8k_records.jsonl")
    for record in records:
        source = train if record["source_split"] == "train" else test
        validate_gsm8k_source_row(record, dict(source[int(record["source_index"])]))
    return {
        "dataset": "gsm8k",
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "source_counts": {"train": len(train), "test": len(test)},
        "frozen_record_count": len(records),
        "validated_record_count": len(records),
    }


def _ood(config_path: Path, dataset: str) -> dict:
    contract = resolve_ood_contract(
        repo_root=ROOT, matrix_config_path=config_path, dataset=dataset
    )
    spec = contract["source"]
    source = load_dataset(
        spec["repo_id"],
        spec["config"],
        split=spec["split"],
        revision=spec["revision"],
    )
    for record in contract["records"]:
        validate_source_row(
            record=record,
            raw_row=dict(source[int(record["source_index"])]),
            answer_field=str(spec["answer_field"]),
        )
    return {
        "dataset": dataset,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "split": spec["split"],
        "source_count": len(source),
        "frozen_record_count": len(contract["records"]),
        "validated_record_count": len(contract["records"]),
        "records_sha256": contract["records_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_clean_common24_v8_canonical.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise ValueError("v8 dataset qualification must run with HF_DATASETS_OFFLINE=1")
    config_path = args.config.resolve()
    rows = [_gsm8k(config_path)] + [
        _ood(config_path, dataset) for dataset in OOD_DATASETS
    ]
    stable = {
        "schema_version": "phase2-v8-offline-dataset-cache-v1",
        "status": "PASS",
        "protocol_id": "phase2-clean-common24-v8",
        "offline_mode": True,
        "datasets": rows,
        "gpu_accessed": False,
    }
    payload = stable | {"dataset_cache_contract_sha256": canonical_json_sha256(stable)}
    write_exclusive_or_verify(args.output.resolve(), canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
