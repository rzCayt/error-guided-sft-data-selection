"""Resolved-contract and tokenized-training evidence for clean Phase-2 v8."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from eg_sft.training.effective_batch import TrainingItem
from eg_sft.training.response_only import IGNORE_INDEX


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def selected_id_order(selection: Mapping[str, Any]) -> list[str]:
    selected = selection.get("selected")
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise ValueError("selection has no ordered selected rows")
    ids = [str(row["candidate_id"]) for row in selected]
    if len(ids) != 500 or len(ids) != len(set(ids)):
        raise ValueError("selection must contain 500 unique ordered candidate IDs")
    return ids


def scientific_static_view(contract: Mapping[str, Any]) -> dict[str, Any]:
    selection = contract["selection"]
    return {
        "method": contract["method"],
        "replicate_index": int(contract["replicate_index"]),
        "selection_manifest_sha256": selection["file_sha256"],
        "selected_id_sha256": selection["selected_id_sha256"],
        "selected_id_order_sha256": canonical_sha256(selected_id_order(selection)),
        "protocol_sha256": canonical_sha256(contract["protocol"]),
        "base_recipe_sha256": canonical_sha256(contract["base_recipe"]),
        "training": contract["config"]["training"],
        "evaluation": contract["config"]["evaluation"],
        "data_manifest": contract["config"]["data_manifest"],
    }


def resolved_contract_evidence(
    *, child: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    child_view = scientific_static_view(child)
    parent_view = scientific_static_view(parent)
    changed = [
        key for key in sorted(set(child_view) | set(parent_view))
        if child_view.get(key) != parent_view.get(key)
    ]
    expected_seed_change = int(child["seed"]) != int(parent["seed"])
    report = {
        "schema_version": "phase2-v8-resolved-contract-diff-v1",
        "status": "PASS" if not changed else "FAIL",
        "child_cell_id": child["cell_id"],
        "parent_cell_id": parent["cell_id"],
        "child_train_seed": int(child["seed"]),
        "parent_train_seed": int(parent["seed"]),
        "train_seed_changed": expected_seed_change,
        "allowed_identity_changes": [
            "cell_id",
            "parent_cell_id",
            "train_seed",
            "output_root",
            "study",
            "attempt_id",
            "worker_id",
            "timestamps",
            "rng_derived_training_order",
        ],
        "unexpected_scientific_changes": changed,
        "child_static_view_sha256": canonical_sha256(child_view),
        "parent_static_view_sha256": canonical_sha256(parent_view),
        "scientific_static_views_equal": not changed,
    }
    if changed:
        raise ValueError(f"parent-child scientific contract changed: {changed}")
    return report


def runtime_training_hashes(
    *,
    cell_id: str,
    train_seed: int,
    selection_manifest_sha256: str,
    selected_ids: Sequence[str],
    tokenized_examples: Sequence[Mapping[str, Sequence[int]]],
    epoch_orders: Sequence[Sequence[int]],
    step_plan: Sequence[Sequence[TrainingItem]],
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    ids = [str(value) for value in selected_ids]
    if len(ids) != len(tokenized_examples):
        raise ValueError("selected IDs and tokenized examples differ in length")
    input_payload = [
        [record_id, [int(value) for value in example["input_ids"]]]
        for record_id, example in zip(ids, tokenized_examples, strict=True)
    ]
    label_payload = [
        [record_id, [int(value) for value in example["labels"]]]
        for record_id, example in zip(ids, tokenized_examples, strict=True)
    ]
    prompt_token_count = sum(
        int(value) == IGNORE_INDEX
        for example in tokenized_examples
        for value in example["labels"]
    )
    non_padding_token_count = sum(
        len(example["input_ids"]) for example in tokenized_examples
    )
    occurrence_payload = [
        [epoch, position, ids[int(example_index)]]
        for epoch, order in enumerate(epoch_orders)
        for position, example_index in enumerate(order)
    ]
    step_payload = [
        [
            [int(item.epoch), int(item.position), ids[int(item.example_index)]]
            for item in step
        ]
        for step in step_plan
    ]
    response_counts = []
    for step in step_plan:
        count = 0
        for item in step:
            labels = tokenized_examples[int(item.example_index)]["labels"]
            count += sum(
                int(value) != IGNORE_INDEX for value in list(labels)[1:]
            )
        response_counts.append(count)
    rng_map = {
        "python_random_seed": int(train_seed),
        "numpy_seed": int(train_seed),
        "torch_cpu_seed": int(train_seed),
        "torch_cuda_seed": int(train_seed),
        "transformers_set_seed": int(train_seed),
        "lora_initialization_seed": int(train_seed),
        "dropout_seed_source": "torch_rng_from_train_seed",
        "dataloader_order_seed": int(train_seed),
    }
    return {
        "schema_version": "phase2-v8-training-input-hashes-v1",
        "status": "PASS",
        "cell_id": cell_id,
        "train_seed": int(train_seed),
        "selection_manifest_sha256": selection_manifest_sha256,
        "selected_count": len(ids),
        "selected_id_set_sha256": canonical_sha256(sorted(ids)),
        "selected_id_order_sha256": canonical_sha256(ids),
        "tokenized_input_sha256": canonical_sha256(input_payload),
        "label_mask_sha256": canonical_sha256(label_payload),
        "ordered_sample_occurrence_sha256": canonical_sha256(occurrence_payload),
        "optimizer_step_plan_sha256": canonical_sha256(step_payload),
        "step_response_token_counts_sha256": canonical_sha256(response_counts),
        "step_response_token_counts": response_counts,
        "optimizer_steps": len(step_plan),
        "response_supervision_exposure_tokens": sum(response_counts),
        "prompt_token_count": prompt_token_count,
        "non_padding_token_count": non_padding_token_count,
        "training_config_sha256": canonical_sha256(training_config),
        "rng_map_sha256": canonical_sha256(rng_map),
        "rng_map": rng_map,
        "gpu_accessed": False,
    }


def validate_runtime_training_hashes(
    *, expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    exact_fields = (
        "cell_id",
        "train_seed",
        "selection_manifest_sha256",
        "selected_count",
        "selected_id_set_sha256",
        "selected_id_order_sha256",
        "tokenized_input_sha256",
        "label_mask_sha256",
        "ordered_sample_occurrence_sha256",
        "optimizer_step_plan_sha256",
        "step_response_token_counts_sha256",
        "step_response_token_counts",
        "optimizer_steps",
        "response_supervision_exposure_tokens",
        "prompt_token_count",
        "non_padding_token_count",
        "training_config_sha256",
        "rng_map_sha256",
    )
    changed = [field for field in exact_fields if expected.get(field) != observed.get(field)]
    if changed:
        raise ValueError(f"runtime training input contract changed: {changed}")
