"""CPU-only evidence builder for the real-tokenizer dose-cap dry run.

This module does not load Hugging Face assets itself.  The command-line entry
point deliberately obtains ``tokenized_examples`` through the same
``_prepare_training_data`` function used by formal training, then passes the
result here for strict, easily unit-tested validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eg_sft.experiment.formal_runtime import deterministic_epoch_orders
from eg_sft.training.token_budget import build_hash_uniform_token_cap_plan


EVIDENCE_TYPE = "real_tokenizer_cpu_dry_run"
SCHEMA_VERSION = "dose-only-real-tokenizer-cpu-dry-run-v1"


def canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one immutable JSON artifact and fail if the target exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def build_real_tokenizer_dry_run_report(
    *,
    cell_id: str,
    selected: Sequence[Mapping[str, Any]],
    tokenized_examples: Sequence[Mapping[str, Sequence[int]]],
    token_audit: Sequence[Mapping[str, Any]],
    epochs: int,
    optimizer_steps: int,
    seed: int,
    supervision_token_cap: int,
    token_cap_policy: str,
    bindings: Mapping[str, Any],
    expected_selected_count: int = 500,
) -> dict[str, Any]:
    """Validate a real-tokenizer cap plan and return claim-bounded evidence."""

    if len(selected) != expected_selected_count:
        raise ValueError("selected candidate count changed")
    if len(tokenized_examples) != len(selected) or len(token_audit) != len(selected):
        raise ValueError("real tokenization rows do not match the frozen selection")

    selected_ids = [str(row["candidate_id"]) for row in selected]
    audited_ids = [str(row["candidate_id"]) for row in token_audit]
    if selected_ids != audited_ids:
        raise ValueError("real token audit order differs from frozen selected IDs")
    if len(set(selected_ids)) != expected_selected_count:
        raise ValueError("selected candidate IDs are not unique")

    for candidate, audit, example in zip(
        selected,
        token_audit,
        tokenized_examples,
        strict=True,
    ):
        actual_total = len(example["input_ids"])
        actual_supervised = sum(int(value) != -100 for value in example["labels"])
        expected_pair = (
            int(candidate["total_tokens"]),
            int(candidate["supervised_tokens"]),
        )
        if (actual_total, actual_supervised) != expected_pair:
            raise ValueError(
                f"real tokenization differs from frozen audit for {candidate['candidate_id']}"
            )
        if (
            int(audit["total_tokens"]),
            int(audit["supervised_tokens"]),
        ) != expected_pair:
            raise ValueError(
                f"formal token audit differs from frozen metadata for {candidate['candidate_id']}"
            )

    epoch_orders = deterministic_epoch_orders(
        example_count=len(tokenized_examples),
        epochs=epochs,
        seed=seed,
    )
    plan = build_hash_uniform_token_cap_plan(
        epoch_orders=epoch_orders,
        tokenized_examples=tokenized_examples,
        record_ids=selected_ids,
        supervision_token_cap=supervision_token_cap,
        optimizer_steps=optimizer_steps,
        seed=seed,
        policy=token_cap_policy,
    )

    per_epoch_candidate_tokens = sum(
        int(row["supervised_tokens"]) for row in token_audit
    )
    formal_candidate_exposure = per_epoch_candidate_tokens * epochs
    planner_candidate_exposure = sum(
        mask.candidate_tokens for mask in plan.step_masks
    )
    per_step_kept = [mask.kept_tokens for mask in plan.step_masks]
    expected_tokens_per_step, remainder = divmod(
        supervision_token_cap,
        optimizer_steps,
    )
    if remainder:
        raise ValueError("supervision token cap is not divisible by optimizer steps")

    occurrence_count = expected_selected_count * epochs
    gates = {
        "selected_id_count_is_500": len(selected_ids) == 500,
        "occurrence_count_is_1000": plan.occurrence_count == 1000,
        "optimizer_step_count_is_64": len(plan.step_masks) == optimizer_steps == 64,
        "each_step_keeps_995": set(per_step_kept) == {995},
        "kept_exposure_is_63680": sum(per_step_kept) == supervision_token_cap == 63680,
        "candidate_exposure_matches_formal_real_tokenization": (
            planner_candidate_exposure == formal_candidate_exposure
        ),
        "all_selected_ids_have_kept_tokens": (
            plan.selected_candidate_id_coverage
            == plan.candidate_id_count
            == expected_selected_count
        ),
        "all_occurrences_have_kept_tokens": (
            plan.occurrence_with_kept_token_count
            == plan.occurrence_count
            == occurrence_count
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"real-tokenizer CPU dry-run gates failed: {failed}")

    cumulative = 0
    step_rows: list[dict[str, Any]] = []
    for step_index, mask in enumerate(plan.step_masks, start=1):
        cumulative += mask.kept_tokens
        step_rows.append(
            {
                "optimizer_step": step_index,
                "candidate_response_supervision_tokens": mask.candidate_tokens,
                "kept_response_supervision_tokens": mask.kept_tokens,
                "cumulative_response_supervision_tokens": cumulative,
                "token_cap_mask_sha256": mask.mask_sha256,
            }
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": EVIDENCE_TYPE,
        "status": "PASS",
        "cell_id": cell_id,
        "bindings": dict(bindings),
        "real_tokenization": {
            "selected_id_count": len(selected_ids),
            "selected_id_order_sha256": canonical_json_sha256(selected_ids),
            "tokenized_example_count": len(tokenized_examples),
            "formal_token_audit_row_count": len(token_audit),
            "candidate_supervision_tokens_per_epoch": per_epoch_candidate_tokens,
            "candidate_supervision_exposure_tokens": formal_candidate_exposure,
            "planner_candidate_supervision_exposure_tokens": planner_candidate_exposure,
            "formal_real_tokenization_match": True,
        },
        "dose_plan": {
            "epochs": epochs,
            "occurrence_count": plan.occurrence_count,
            "optimizer_steps": optimizer_steps,
            "tokens_per_optimizer_step": expected_tokens_per_step,
            "supervision_token_cap": supervision_token_cap,
            "token_cap_policy": token_cap_policy,
            "kept_supervision_exposure_tokens": cumulative,
            "selected_token_set_sha256": plan.selected_token_set_sha256,
            "boundary_split_occurrence_count": plan.boundary_split_occurrence_count,
            "selected_candidate_id_coverage": plan.selected_candidate_id_coverage,
            "candidate_id_count": plan.candidate_id_count,
            "occurrence_with_kept_token_count": plan.occurrence_with_kept_token_count,
            "mandatory_coverage_token_count": plan.mandatory_coverage_token_count,
            "steps": step_rows,
        },
        "gates": gates,
        "prior_metadata_artifacts": [
            {
                "path": "results/cpu_identifiability_audit/dose_only_token_cap_feasibility.json",
                "evidence_type": "metadata_simulation",
            },
            {
                "path": "results/cpu_identifiability_audit/dose_only_token_cap_feasibility_v2.json",
                "evidence_type": "metadata_simulation",
            },
            {
                "path": (
                    "results/cpu_identifiability_audit/"
                    "dose_only_token_cap_coverage_feasibility_v2.json"
                ),
                "evidence_type": "metadata_simulation",
            },
        ],
        "claim_boundary": (
            "This PASS proves that the frozen rep1 selection can be tokenized on CPU "
            "through the formal response-only data path and can produce a deterministic "
            "63,680-token coverage-constrained plan. It is not evidence that GPU training, "
            "checkpointing, optimization, or downstream evaluation succeeds."
        ),
    }
    report["artifact_content_sha256"] = canonical_json_sha256(report)
    return report
