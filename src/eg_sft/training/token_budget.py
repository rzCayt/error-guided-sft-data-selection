"""Optimizer-step planning for a fixed response-token exposure protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from eg_sft.training.effective_batch import TrainingItem
from eg_sft.training.response_only import IGNORE_INDEX


HASH_UNIFORM_TOKEN_CAP_POLICY = "hash_uniform_v1"


@dataclass(frozen=True)
class StepSupervisionMask:
    """One deterministic response-token mask for an optimizer step."""

    candidate_tokens: int
    kept_tokens: int
    mask_sha256: str
    selected_token_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TokenCapTrainingPlan:
    """A 64-step token-dose plan over the frozen occurrence order."""

    step_items: tuple[tuple[TrainingItem, ...], ...]
    step_masks: tuple[StepSupervisionMask, ...]
    selected_token_set_sha256: str
    boundary_split_occurrence_count: int
    selected_candidate_id_coverage: int
    candidate_id_count: int
    occurrence_with_kept_token_count: int
    occurrence_count: int
    mandatory_coverage_token_count: int


def supervision_tokens_per_step(
    *,
    supervision_token_cap: int,
    optimizer_steps: int,
    policy: str,
) -> int:
    """Validate an equal-per-step supervision cap and return its step target."""

    if policy != HASH_UNIFORM_TOKEN_CAP_POLICY:
        raise ValueError(f"unsupported supervision token-cap policy: {policy}")
    if supervision_token_cap <= 0:
        raise ValueError("supervision_token_cap must be positive")
    if optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be positive")
    per_step, remainder = divmod(supervision_token_cap, optimizer_steps)
    if remainder:
        raise ValueError(
            "supervision_token_cap must be divisible by optimizer_steps for "
            "an equal-per-step dose"
        )
    return per_step


def resolve_token_cap_options(
    *,
    cli_supervision_token_cap: int | None,
    cli_token_cap_policy: str | None,
    contract_supervision_token_cap: int | None,
    contract_token_cap_policy: str | None,
) -> tuple[int | None, str | None]:
    """Resolve token-cap options while preventing frozen-contract overrides."""

    contract_pair = (contract_supervision_token_cap, contract_token_cap_policy)
    cli_pair = (cli_supervision_token_cap, cli_token_cap_policy)
    if (contract_pair[0] is None) != (contract_pair[1] is None):
        raise ValueError("contract supervision token cap and policy must be paired")
    if contract_pair[0] is not None:
        frozen = (int(contract_pair[0]), str(contract_pair[1]))
        if cli_pair == (None, None):
            return frozen
        if cli_pair[0] is None or cli_pair[1] is None:
            raise ValueError("CLI token cap and policy must be provided together")
        requested = (int(cli_pair[0]), str(cli_pair[1]))
        if requested != frozen:
            raise ValueError("CLI token-cap options differ from the frozen contract")
        return frozen
    if cli_pair == (None, None):
        return None, None
    if cli_pair[0] is None or cli_pair[1] is None:
        raise ValueError("CLI token cap and policy must be provided together")
    return int(cli_pair[0]), str(cli_pair[1])


def _token_priority(
    *, record_id: str, epoch: int, token_index: int, seed: int
) -> bytes:
    identity = json.dumps(
        [record_id, epoch, token_index, seed],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).digest()


def build_hash_uniform_step_mask(
    *,
    step_items: Sequence[TrainingItem],
    tokenized_examples: Sequence[Mapping[str, Sequence[int]]],
    record_ids: Sequence[str],
    kept_tokens: int,
    seed: int,
) -> StepSupervisionMask:
    """Choose exactly ``kept_tokens`` by a stable token-identity hash.

    Legacy test-only prototype.  The formal dose-only runtime does not call
    this function; it calls ``build_hash_uniform_token_cap_plan`` so the global
    token dose and equal optimizer-step boundaries are constructed together.
    This helper is retained only to document the rejected per-step design.

    Hashing ``(record_id, epoch, token_index, seed)`` makes the decision
    independent of method labels and avoids contiguous tail truncation. A
    token at label position zero is excluded because causal loss shifts labels
    left once and therefore cannot contribute to the optimized objective.
    """

    if kept_tokens <= 0:
        raise ValueError("kept_tokens must be positive")
    if len(tokenized_examples) != len(record_ids):
        raise ValueError("tokenized_examples and record_ids must have equal lengths")
    if len(set(map(str, record_ids))) != len(record_ids):
        raise ValueError("record_ids must be unique for occurrence coverage")
    if not step_items:
        raise ValueError("step_items must not be empty")

    candidates: list[tuple[bytes, int, int, str, int]] = []
    for item_slot, item in enumerate(step_items):
        if item.example_index < 0 or item.example_index >= len(tokenized_examples):
            raise ValueError("training item example_index is out of range")
        labels = tokenized_examples[item.example_index].get("labels")
        if labels is None:
            raise ValueError("tokenized example has no labels")
        record_id = str(record_ids[item.example_index])
        if not record_id:
            raise ValueError("record IDs must be non-empty")
        for token_index, label in enumerate(labels):
            if token_index == 0 or int(label) == IGNORE_INDEX:
                continue
            candidates.append(
                (
                    _token_priority(
                        record_id=record_id,
                        epoch=int(item.epoch),
                        token_index=token_index,
                        seed=seed,
                    ),
                    item_slot,
                    token_index,
                    record_id,
                    int(item.epoch),
                )
            )
    if len(candidates) < kept_tokens:
        raise ValueError(
            "optimizer step has fewer candidate supervised tokens than its token cap"
        )

    ranked = sorted(
        candidates,
        key=lambda row: (row[0], row[3], row[4], row[2], row[1]),
    )
    selected = {(row[1], row[2]) for row in ranked[:kept_tokens]}
    selected_by_item = tuple(
        tuple(
            token_index
            for token_index, label in enumerate(
                tokenized_examples[item.example_index]["labels"]
            )
            if token_index > 0
            and int(label) != IGNORE_INDEX
            and (item_slot, token_index) in selected
        )
        for item_slot, item in enumerate(step_items)
    )
    if sum(map(len, selected_by_item)) != kept_tokens:  # pragma: no cover
        raise AssertionError("hash-uniform mask lost selected tokens")

    mask_payload = [
        [record_id, epoch, token_index, int((item_slot, token_index) in selected)]
        for _, item_slot, token_index, record_id, epoch in candidates
    ]
    mask_text = json.dumps(
        mask_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return StepSupervisionMask(
        candidate_tokens=len(candidates),
        kept_tokens=kept_tokens,
        mask_sha256=hashlib.sha256(mask_text.encode("utf-8")).hexdigest(),
        selected_token_indices=selected_by_item,
    )


def build_hash_uniform_token_cap_plan(
    *,
    epoch_orders: Sequence[Sequence[int]],
    tokenized_examples: Sequence[Mapping[str, Sequence[int]]],
    record_ids: Sequence[str],
    supervision_token_cap: int,
    optimizer_steps: int,
    seed: int,
    policy: str = HASH_UNIFORM_TOKEN_CAP_POLICY,
) -> TokenCapTrainingPlan:
    """Hash-sample one global dose, then split it into equal ordered steps.

    Response lengths vary enough that the legacy 15/16-example step groups can
    contain fewer tokens than an equal token target.  This planner therefore
    selects the global token dose first, traverses the frozen occurrence/token
    order, and creates a new optimizer boundary after every equal token block.
    A boundary occurrence may appear in two consecutive steps with disjoint
    masks; selected IDs, occurrence order, seed, step count and scheduler are
    unchanged.
    """

    kept_per_step = supervision_tokens_per_step(
        supervision_token_cap=supervision_token_cap,
        optimizer_steps=optimizer_steps,
        policy=policy,
    )
    if len(tokenized_examples) != len(record_ids):
        raise ValueError("tokenized_examples and record_ids must have equal lengths")
    occurrences = [
        TrainingItem(epoch=epoch, position=position, example_index=int(example_index))
        for epoch, order in enumerate(epoch_orders)
        for position, example_index in enumerate(order)
    ]
    if not occurrences:
        raise ValueError("epoch_orders must contain at least one occurrence")

    # Stream order is fixed by the original multi-epoch example order.  Hash
    # priority determines only whether each supervised token is retained.
    candidates: list[tuple[bytes, TrainingItem, int, str]] = []
    for item in occurrences:
        if item.example_index < 0 or item.example_index >= len(tokenized_examples):
            raise ValueError("training item example_index is out of range")
        labels = tokenized_examples[item.example_index].get("labels")
        if labels is None:
            raise ValueError("tokenized example has no labels")
        record_id = str(record_ids[item.example_index])
        if not record_id:
            raise ValueError("record IDs must be non-empty")
        for token_index, label in enumerate(labels):
            if token_index == 0 or int(label) == IGNORE_INDEX:
                continue
            candidates.append(
                (
                    _token_priority(
                        record_id=record_id,
                        epoch=int(item.epoch),
                        token_index=token_index,
                        seed=seed,
                    ),
                    item,
                    token_index,
                    record_id,
                )
            )
    if len(candidates) < supervision_token_cap:
        raise ValueError("training data has fewer supervised tokens than the token cap")
    def priority_key(
        row: tuple[bytes, TrainingItem, int, str]
    ) -> tuple[bytes, str, int, int, int]:
        return (
            row[0],
            row[3],
            row[1].epoch,
            row[2],
            row[1].position,
        )

    # Preserve effective selected-ID and occurrence coverage.  Each occurrence
    # contributes its lowest-hash supervised token before the remaining global
    # dose is filled by the same hash ranking.
    mandatory_by_occurrence: dict[
        TrainingItem, tuple[bytes, TrainingItem, int, str]
    ] = {}
    for row in candidates:
        current = mandatory_by_occurrence.get(row[1])
        if current is None or priority_key(row) < priority_key(current):
            mandatory_by_occurrence[row[1]] = row
    if supervision_token_cap < len(mandatory_by_occurrence):
        raise ValueError("token cap is too small to preserve occurrence coverage")
    mandatory = {
        (row[1], row[2]) for row in mandatory_by_occurrence.values()
    }
    ranked_remaining = sorted(
        (row for row in candidates if (row[1], row[2]) not in mandatory),
        key=priority_key,
    )
    selected = mandatory | {
        (row[1], row[2])
        for row in ranked_remaining[
            : supervision_token_cap - len(mandatory)
        ]
    }
    if len(selected) != supervision_token_cap:  # pragma: no cover
        raise AssertionError("coverage-constrained hash selection has the wrong size")

    segment_candidates: list[list[tuple[TrainingItem, int, str, bool]]] = [
        [] for _ in range(optimizer_steps)
    ]
    segment_selected: list[dict[TrainingItem, list[int]]] = [
        {} for _ in range(optimizer_steps)
    ]
    step_index = 0
    kept_in_step = 0
    for _, item, token_index, record_id in candidates:
        is_kept = (item, token_index) in selected
        if is_kept and kept_in_step == kept_per_step:
            step_index += 1
            kept_in_step = 0
        if step_index >= optimizer_steps:
            # Only unselected tail tokens may remain after the last selected
            # token. They belong to the final mask candidate universe.
            if is_kept:  # pragma: no cover - guarded by exact cap arithmetic
                raise AssertionError("token cap produced too many optimizer steps")
            assigned_step = optimizer_steps - 1
        else:
            assigned_step = step_index
        segment_candidates[assigned_step].append(
            (item, token_index, record_id, is_kept)
        )
        if is_kept:
            segment_selected[assigned_step].setdefault(item, []).append(token_index)
            kept_in_step += 1

    step_items: list[tuple[TrainingItem, ...]] = []
    step_masks: list[StepSupervisionMask] = []
    for index in range(optimizer_steps):
        selected_by_item = segment_selected[index]
        items = tuple(selected_by_item)
        indices = tuple(tuple(selected_by_item[item]) for item in items)
        kept = sum(map(len, indices))
        if kept != kept_per_step:
            raise AssertionError(
                f"token-cap step {index + 1} kept {kept}, expected {kept_per_step}"
            )
        mask_payload = [
            [
                record_id,
                int(item.epoch),
                token_index,
                int(is_kept),
            ]
            for item, token_index, record_id, is_kept in segment_candidates[index]
        ]
        mask_text = json.dumps(
            mask_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        step_items.append(items)
        step_masks.append(
            StepSupervisionMask(
                candidate_tokens=len(segment_candidates[index]),
                kept_tokens=kept,
                mask_sha256=hashlib.sha256(mask_text.encode("utf-8")).hexdigest(),
                selected_token_indices=indices,
            )
        )
    if sum(mask.candidate_tokens for mask in step_masks) != len(candidates):
        raise AssertionError("token-cap plan lost candidate tokens")
    appearances: dict[TrainingItem, int] = {}
    for items in step_items:
        for item in items:
            appearances[item] = appearances.get(item, 0) + 1
    selected_payload = [
        [record_id, int(item.epoch), token_index]
        for _, item, token_index, record_id in candidates
        if (item, token_index) in selected
    ]
    selected_text = json.dumps(
        selected_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    covered_occurrences = {item for item, _ in selected}
    covered_ids = {str(record_ids[item.example_index]) for item in covered_occurrences}
    if len(covered_occurrences) != len(occurrences):  # pragma: no cover
        raise AssertionError("token-cap plan did not preserve every occurrence")
    if len(covered_ids) != len(record_ids):  # pragma: no cover
        raise AssertionError("token-cap plan did not preserve every selected candidate ID")
    return TokenCapTrainingPlan(
        step_items=tuple(step_items),
        step_masks=tuple(step_masks),
        selected_token_set_sha256=hashlib.sha256(
            selected_text.encode("utf-8")
        ).hexdigest(),
        boundary_split_occurrence_count=sum(value > 1 for value in appearances.values()),
        selected_candidate_id_coverage=len(covered_ids),
        candidate_id_count=len(record_ids),
        occurrence_with_kept_token_count=len(covered_occurrences),
        occurrence_count=len(occurrences),
        mandatory_coverage_token_count=len(mandatory),
    )


def apply_supervision_mask(
    example: Mapping[str, Sequence[int]], *, selected_token_indices: Sequence[int]
) -> dict[str, list[int]]:
    """Return a copy with every non-selected response label ignored."""

    labels = [int(value) for value in example["labels"]]
    selected = {int(index) for index in selected_token_indices}
    if any(index <= 0 or index >= len(labels) for index in selected):
        raise ValueError("selected token index is outside the shifted-label domain")
    if any(labels[index] == IGNORE_INDEX for index in selected):
        raise ValueError("selected token index points to an ignored label")
    masked_labels = [
        label if index in selected else IGNORE_INDEX
        for index, label in enumerate(labels)
    ]
    if sum(label != IGNORE_INDEX for label in masked_labels) != len(selected):
        raise AssertionError("supervision mask kept an unexpected token count")
    return {
        key: ([int(value) for value in values] if key != "labels" else masked_labels)
        for key, values in example.items()
    }


def balanced_optimizer_step_plan(
    *,
    epoch_orders: Sequence[Sequence[int]],
    optimizer_steps: int,
) -> list[tuple[TrainingItem, ...]]:
    """Partition all example occurrences into exactly ``optimizer_steps`` groups.

    The groups preserve the frozen epoch order and differ in sequence count by
    at most one. Losses inside a group must be summed over response tokens and
    normalized once by that group's response-token count.
    """

    if optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be positive")
    flattened = [
        TrainingItem(epoch=epoch, position=position, example_index=int(example_index))
        for epoch, order in enumerate(epoch_orders)
        for position, example_index in enumerate(order)
    ]
    if len(flattened) < optimizer_steps:
        raise ValueError("optimizer_steps cannot exceed example occurrences")
    base, remainder = divmod(len(flattened), optimizer_steps)
    groups: list[tuple[TrainingItem, ...]] = []
    cursor = 0
    for step_index in range(optimizer_steps):
        size = base + (1 if step_index < remainder else 0)
        groups.append(tuple(flattened[cursor : cursor + size]))
        cursor += size
    if cursor != len(flattened):  # pragma: no cover - defensive invariant
        raise AssertionError("optimizer step plan lost example occurrences")
    return groups


def micro_batches_for_step(
    step_items: Sequence[TrainingItem], *, micro_batch_size: int
) -> list[tuple[TrainingItem, ...]]:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if not step_items:
        raise ValueError("optimizer step cannot be empty")
    return [
        tuple(step_items[start : start + micro_batch_size])
        for start in range(0, len(step_items), micro_batch_size)
    ]


def optimizer_step_token_audit(
    *,
    step_token_counts: Sequence[int],
    expected_optimizer_steps: int,
    expected_exposure_tokens: int,
    tolerance_fraction: float,
) -> dict[str, float | int | bool]:
    if len(step_token_counts) != expected_optimizer_steps:
        raise ValueError("optimizer step token count length differs from protocol")
    if any(value <= 0 for value in step_token_counts):
        raise ValueError("each optimizer step must contain supervised response tokens")
    observed = sum(step_token_counts)
    relative_error = abs(observed - expected_exposure_tokens) / expected_exposure_tokens
    return {
        "optimizer_steps": len(step_token_counts),
        "response_supervision_exposure_tokens": observed,
        "target_response_supervision_exposure_tokens": expected_exposure_tokens,
        "exposure_relative_error": relative_error,
        "exposure_gate_passed": relative_error <= tolerance_fraction,
        "minimum_step_response_tokens": min(step_token_counts),
        "maximum_step_response_tokens": max(step_token_counts),
    }
