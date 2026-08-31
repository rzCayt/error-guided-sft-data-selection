from __future__ import annotations

import copy
from pathlib import Path

import pytest

from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract
from eg_sft.experiment.formal_runtime import deterministic_epoch_orders
from eg_sft.experiment.phase2_v8_contract_audit import (
    resolved_contract_evidence,
    runtime_training_hashes,
    validate_runtime_training_hashes,
)
from eg_sft.training.token_budget import balanced_optimizer_step_plan


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "configs" / "phase2_clean_common24_v8_canonical.json"
PARENT = ROOT / "configs" / "budget_equivalent_phase1_matrix_frozen_20260824_v2.json"


def test_real_parent_child_scientific_contract_diff_is_empty() -> None:
    child = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=V8,
        cell_id="v8_rep1_random_common_mix_train29",
    )
    parent = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=PARENT,
        cell_id="rep1_random_common_mix_train17",
    )
    report = resolved_contract_evidence(child=child, parent=parent)
    assert report["status"] == "PASS"
    assert report["train_seed_changed"] is True
    assert report["unexpected_scientific_changes"] == []


def test_runtime_hashes_bind_order_labels_steps_and_rng() -> None:
    ids = ["a", "b"]
    examples = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ]
    orders = deterministic_epoch_orders(example_count=2, epochs=2, seed=17)
    plan = balanced_optimizer_step_plan(epoch_orders=orders, optimizer_steps=2)
    observed = runtime_training_hashes(
        cell_id="cell",
        train_seed=17,
        selection_manifest_sha256="a" * 64,
        selected_ids=ids,
        tokenized_examples=examples,
        epoch_orders=orders,
        step_plan=plan,
        training_config={"optimizer_steps": 2},
    )
    validate_runtime_training_hashes(expected=observed, observed=observed)
    broken = copy.deepcopy(observed)
    broken["label_mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="label_mask_sha256"):
        validate_runtime_training_hashes(expected=observed, observed=broken)


def test_same_seed_reproduces_occurrence_and_step_hashes() -> None:
    examples = [
        {"input_ids": [index, 1], "labels": [-100, 1]} for index in range(4)
    ]
    def evidence(seed: int) -> dict:
        orders = deterministic_epoch_orders(example_count=4, epochs=2, seed=seed)
        plan = balanced_optimizer_step_plan(epoch_orders=orders, optimizer_steps=4)
        return runtime_training_hashes(
            cell_id=f"cell-{seed}",
            train_seed=seed,
            selection_manifest_sha256="b" * 64,
            selected_ids=[f"r{i}" for i in range(4)],
            tokenized_examples=examples,
            epoch_orders=orders,
            step_plan=plan,
            training_config={"optimizer_steps": 4},
        )
    left = evidence(17)
    right = evidence(17)
    changed = evidence(29)
    assert left["ordered_sample_occurrence_sha256"] == right[
        "ordered_sample_occurrence_sha256"
    ]
    assert left["optimizer_step_plan_sha256"] != changed[
        "optimizer_step_plan_sha256"
    ]
    assert left["tokenized_input_sha256"] == changed["tokenized_input_sha256"]
    assert left["label_mask_sha256"] == changed["label_mask_sha256"]
    assert left["selected_id_set_sha256"] == changed["selected_id_set_sha256"]
    assert left["prompt_token_count"] == changed["prompt_token_count"]
    assert left["non_padding_token_count"] == changed["non_padding_token_count"]
