"""CPU-only materialization of all 24 resolved and tokenized v8 contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    read_json_object,
    resolve_phase1_contract,
)
from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.experiment.formal_runtime import deterministic_epoch_orders  # noqa: E402
from eg_sft.experiment.phase2_v8_contract_audit import (  # noqa: E402
    resolved_contract_evidence,
    runtime_training_hashes,
    scientific_static_view,
    selected_id_order,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402
from eg_sft.training.token_budget import balanced_optimizer_step_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_clean_common24_v8_canonical.json")
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from run_b500_formal_resumable import _prepare_training_data
    from run_budget_equivalent_cell import _resolved_recipe

    config_path = args.config.resolve()
    config = read_json_object(config_path)
    tokenizer_path = args.tokenizer_path.resolve(strict=True)
    required_tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if any(not (tokenizer_path / name).is_file() for name in required_tokenizer_files):
        raise ValueError("v8 contract materialization tokenizer is incomplete")
    # The archived Qwen2 tokenizer directory intentionally has no model config.
    # Transformers >=4.57 otherwise emits a Mistral-specific regex warning for
    # any local tokenizer-only directory. Explicit False preserves the frozen
    # Qwen2 behavior and prevents an accidental Mistral rewrite.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=True,
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    output_root = args.output_root.resolve()
    summaries = []
    for job in config["job_order"]:
        child = resolve_phase1_contract(
            repo_root=ROOT, config_path=config_path, cell_id=str(job["cell_id"])
        )
        parent = resolve_phase1_contract(
            repo_root=ROOT,
            config_path=(ROOT / config["parent_matrix"]["path"]).resolve(),
            cell_id=str(job["parent_cell_id"]),
        )
        diff = resolved_contract_evidence(child=child, parent=parent)
        recipe = _resolved_recipe(child)
        examples, _, _, _ = _prepare_training_data(
            protocol=child["protocol"],
            recipe=recipe,
            selected=child["selection"]["selected"],
            data_manifest_dir=child["data_dir"],
            tokenizer=tokenizer,
        )
        ids = selected_id_order(child["selection"])
        orders = deterministic_epoch_orders(
            example_count=len(examples),
            epochs=int(recipe["training"]["epochs"]),
            seed=int(child["seed"]),
        )
        plan = balanced_optimizer_step_plan(
            epoch_orders=orders,
            optimizer_steps=int(recipe["training"]["optimizer_steps"]),
        )
        hashes = runtime_training_hashes(
            cell_id=child["cell_id"],
            train_seed=int(child["seed"]),
            selection_manifest_sha256=child["selection"]["file_sha256"],
            selected_ids=ids,
            tokenized_examples=examples,
            epoch_orders=orders,
            step_plan=plan,
            training_config=recipe["training"],
        )
        cell_root = output_root / child["cell_id"]
        files = {
            "resolved_contract.json": {
                "schema_version": "phase2-v8-resolved-contract-v1",
                "cell_id": child["cell_id"],
                "train_seed": child["seed"],
                "study": child["study"],
                "static_view": scientific_static_view(child),
            },
            "parent_resolved_contract.json": {
                "schema_version": "phase2-v8-parent-resolved-contract-v1",
                "cell_id": parent["cell_id"],
                "train_seed": parent["seed"],
                "study": parent.get("study", "historical_parent"),
                "static_view": scientific_static_view(parent),
            },
            "contract_diff.json": diff,
            "training_input_hashes.json": hashes,
        }
        for name, payload in files.items():
            write_bytes_exclusive_or_verify(
                cell_root / name, canonical_json_bytes(payload)
            )
        summaries.append(
            {
                "cell_id": child["cell_id"],
                "contract_diff_sha256": file_sha256(cell_root / "contract_diff.json"),
                "training_input_hashes_sha256": file_sha256(
                    cell_root / "training_input_hashes.json"
                ),
            }
        )
    tokenizer_files = [
        {"name": name, "sha256": file_sha256(tokenizer_path / name)}
        for name in required_tokenizer_files
    ]
    manifest = {
        "schema_version": "phase2-v8-materialized-contracts-v1",
        "status": "PASS",
        "config_sha256": file_sha256(config_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_source_kind": "archived_qwen2_tokenizer_snapshot",
        "mistral_regex_fix_applied": False,
        "tokenizer_files": tokenizer_files,
        "cell_count": len(summaries),
        "cells": summaries,
        "gpu_accessed": False,
    }
    write_bytes_exclusive_or_verify(
        output_root / "MATERIALIZATION_COMPLETE.json", canonical_json_bytes(manifest)
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
