"""CPU-only feasibility audit for the four frozen v4 dose-only cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.formal_runtime import deterministic_epoch_orders  # noqa: E402
from eg_sft.experiment.identifiable_budget_v4 import (  # noqa: E402
    resolve_identifiable_contract,
)
from eg_sft.training.token_budget import (  # noqa: E402
    build_hash_uniform_token_cap_plan,
)


def _metadata_tokenized_examples(
    selected: list[dict[str, Any]],
) -> list[dict[str, list[int]]]:
    examples = []
    for row in selected:
        total = int(row["total_tokens"])
        supervised = int(row["supervised_tokens"])
        prompt = total - supervised
        if prompt <= 0 or supervised <= 0:
            raise ValueError("selected token metadata is invalid")
        examples.append(
            {
                "input_ids": [0] * total,
                "attention_mask": [1] * total,
                "labels": [-100] * prompt + [1] * supervised,
            }
        )
    return examples


def audit_cell(*, config_path: Path, cell_id: str) -> dict[str, Any]:
    contract = resolve_identifiable_contract(
        repo_root=ROOT,
        config_path=config_path,
        cell_id=cell_id,
    )
    if contract["study"] != "dose_only":
        raise ValueError("token-cap feasibility audit only accepts dose-only cells")
    selected = contract["selection"]["selected"]
    examples = _metadata_tokenized_examples(selected)
    epochs = int(contract["config"]["training"]["epochs"])
    optimizer_steps = int(contract["config"]["training"]["optimizer_steps"])
    orders = deterministic_epoch_orders(
        example_count=len(examples),
        epochs=epochs,
        seed=int(contract["seed"]),
    )
    plan = build_hash_uniform_token_cap_plan(
        epoch_orders=orders,
        tokenized_examples=examples,
        record_ids=[str(row["candidate_id"]) for row in selected],
        supervision_token_cap=int(contract["supervision_token_cap"]),
        optimizer_steps=optimizer_steps,
        seed=int(contract["seed"]),
        policy=str(contract["token_cap_policy"]),
    )

    selected_identities = [
        (item.epoch, item.position, item.example_index, token_index)
        for items, mask in zip(plan.step_items, plan.step_masks, strict=True)
        for item, indices in zip(items, mask.selected_token_indices, strict=True)
        for token_index in indices
    ]
    selected_set = set(selected_identities)
    expected_cap = int(contract["supervision_token_cap"])
    candidate_tokens = sum(mask.candidate_tokens for mask in plan.step_masks)
    expected_candidates = sum(
        int(row["supervised_tokens"]) for row in selected
    ) * epochs
    occurrence_ordinals = [
        item.epoch * len(examples) + item.position
        for items in plan.step_items
        for item in items
    ]
    order_preserved = all(
        left <= right
        for left, right in zip(
            occurrence_ordinals,
            occurrence_ordinals[1:],
            strict=False,
        )
    )
    final_total = len(examples) * epochs
    final_kept = sum(
        (
            epoch,
            position,
            int(example_index),
            len(examples[int(example_index)]["labels"]) - 1,
        )
        in selected_set
        for epoch, order in enumerate(orders)
        for position, example_index in enumerate(order)
    )
    overall_retention = expected_cap / expected_candidates
    final_retention = final_kept / final_total
    final_gap = final_retention - overall_retention
    per_step_kept = [mask.kept_tokens for mask in plan.step_masks]
    per_step_candidates = [mask.candidate_tokens for mask in plan.step_masks]
    passed = (
        len(plan.step_masks) == optimizer_steps == 64
        and set(per_step_kept) == {995}
        and len(selected_identities) == len(selected_set) == expected_cap == 63680
        and candidate_tokens == expected_candidates
        and min(per_step_candidates) >= 995
        and plan.selected_candidate_id_coverage == plan.candidate_id_count == 500
        and plan.occurrence_with_kept_token_count == plan.occurrence_count == 1000
        and plan.mandatory_coverage_token_count == 1000
        and order_preserved
        and abs(final_gap) <= 0.02
    )
    if not passed:
        raise ValueError(f"dose-only token-cap feasibility failed for {cell_id}")
    return {
        "status": "PASS",
        "cell_id": cell_id,
        "selected_id_sha256": contract["selection"]["selected_id_sha256"],
        "train_seed": int(contract["seed"]),
        "optimizer_steps": optimizer_steps,
        "tokens_per_optimizer_step": 995,
        "supervision_token_cap": expected_cap,
        "candidate_supervision_tokens": candidate_tokens,
        "dropped_supervision_tokens": candidate_tokens - expected_cap,
        "minimum_step_candidate_tokens": min(per_step_candidates),
        "maximum_step_candidate_tokens": max(per_step_candidates),
        "selected_token_identities_unique": True,
        "selected_candidate_id_coverage": plan.selected_candidate_id_coverage,
        "candidate_id_count": plan.candidate_id_count,
        "occurrence_with_at_least_one_kept_token": (
            plan.occurrence_with_kept_token_count
        ),
        "occurrence_count": plan.occurrence_count,
        "mandatory_coverage_token_count": plan.mandatory_coverage_token_count,
        "frozen_occurrence_order_preserved": True,
        "legacy_sequence_step_boundaries_preserved": False,
        "boundary_split_occurrence_count": plan.boundary_split_occurrence_count,
        "selected_token_set_sha256": plan.selected_token_set_sha256,
        "final_position_retention_rate": final_retention,
        "overall_token_retention_rate": overall_retention,
        "final_minus_overall_retention_rate": final_gap,
        "final_position_excess_drop_gate_absolute_fraction": 0.02,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/identifiable_budget_v4_matrix.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cell_ids = [
        str(row["cell_id"])
        for row in payload["job_order"]
        if row["study"] == "dose_only"
    ]
    rows = [audit_cell(config_path=config_path, cell_id=cell_id) for cell_id in cell_ids]
    result = {
        "audit_schema_version": "dose-only-token-cap-feasibility-v2",
        "status": "PASS",
        "cell_count": len(rows),
        "cells": rows,
        "claim_boundary": (
            "Coverage-constrained hash_uniform_v1 first keeps the minimum-hash token "
            "from every frozen occurrence, then fills the remaining global dose by the "
            "same hash ranking. Legacy sequence step boundaries are intentionally not "
            "preserved, and boundary occurrences may be forwarded in two consecutive "
            "optimizer steps with disjoint masks."
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
