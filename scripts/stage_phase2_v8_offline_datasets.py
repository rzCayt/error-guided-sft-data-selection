"""Pre-stage every pinned formal-evaluation dataset before offline qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    write_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract  # noqa: E402
from eg_sft.experiment.budget_equivalent_ood_runtime import (  # noqa: E402
    OOD_DATASETS,
    resolve_ood_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_clean_common24_v8_canonical.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("HF_DATASETS_OFFLINE") == "1":
        raise ValueError("dataset staging requires online/cache-fill mode")
    config_path = args.config.resolve()
    contract = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=config_path,
        cell_id="v8_rep1_random_common_mix_train17",
    )
    gsm8k = contract["protocol"]["datasets"]["gsm8k"]
    rows = []
    for split in ("train", "test"):
        dataset = load_dataset(
            gsm8k["repo_id"],
            gsm8k["config"],
            split=split,
            revision=gsm8k["revision"],
        )
        rows.append(
            {
                "dataset": "gsm8k",
                "split": split,
                "repo_id": gsm8k["repo_id"],
                "revision": gsm8k["revision"],
                "row_count": len(dataset),
            }
        )
    for name in OOD_DATASETS:
        ood = resolve_ood_contract(
            repo_root=ROOT, matrix_config_path=config_path, dataset=name
        )
        spec = ood["source"]
        dataset = load_dataset(
            spec["repo_id"],
            spec["config"],
            split=spec["split"],
            revision=spec["revision"],
        )
        rows.append(
            {
                "dataset": name,
                "split": spec["split"],
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "row_count": len(dataset),
            }
        )
    payload = {
        "schema_version": "phase2-v8-dataset-cache-stage-v1",
        "status": "STAGED_REQUIRES_OFFLINE_QUALIFICATION",
        "protocol_id": "phase2-clean-common24-v8",
        "datasets": rows,
        "network_permitted_during_staging_only": True,
        "formal_run_authorized": False,
        "gpu_accessed": False,
    }
    write_exclusive_or_verify(args.output.resolve(), canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
