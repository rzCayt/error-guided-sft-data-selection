import pytest

from eg_sft.training.token_budget import (
    apply_supervision_mask,
    balanced_optimizer_step_plan,
    build_hash_uniform_step_mask,
    build_hash_uniform_token_cap_plan,
    resolve_token_cap_options,
    supervision_tokens_per_step,
)


def test_phase1_step_partition_has_sixty_four_boundaries() -> None:
    epoch_orders = [list(range(500)), list(reversed(range(500)))]
    plan = balanced_optimizer_step_plan(
        epoch_orders=epoch_orders,
        optimizer_steps=64,
    )
    assert len(plan) == 64
    assert sum(map(len, plan)) == 1000


def test_hash_uniform_cap_keeps_exactly_995_tokens_per_step() -> None:
    examples = [
        {
            "input_ids": list(range(71)),
            "attention_mask": [1] * 71,
            "labels": [-100] + list(range(1, 71)),
        }
        for _ in range(500)
    ]
    record_ids = [f"candidate-{index:03d}" for index in range(500)]
    epoch_orders = [list(range(500)), list(reversed(range(500)))]
    kept_per_step = supervision_tokens_per_step(
        supervision_token_cap=63680,
        optimizer_steps=64,
        policy="hash_uniform_v1",
    )
    plan = build_hash_uniform_token_cap_plan(
        epoch_orders=epoch_orders,
        tokenized_examples=examples,
        record_ids=record_ids,
        supervision_token_cap=63680,
        optimizer_steps=64,
        seed=17,
    )
    masks = plan.step_masks

    assert kept_per_step == 995
    assert len(masks) == 64
    assert {mask.kept_tokens for mask in masks} == {995}
    assert sum(mask.kept_tokens for mask in masks) == 63680
    assert sum(mask.candidate_tokens for mask in masks) == 70000
    assert all(mask.candidate_tokens >= mask.kept_tokens for mask in masks)
    assert all(len(mask.mask_sha256) == 64 for mask in masks)

    # Applying masks does not mutate the frozen tokenized examples and retains
    # exactly the positions described by each immutable step mask.
    first_items = plan.step_items[0]
    first_mask = masks[0]
    masked = [
        apply_supervision_mask(
            examples[item.example_index],
            selected_token_indices=indices,
        )
        for item, indices in zip(
            first_items, first_mask.selected_token_indices, strict=True
        )
    ]
    assert sum(label != -100 for row in masked for label in row["labels"]) == 995
    assert all(example["labels"][1:] == list(range(1, 71)) for example in examples)

    selected_identities = [
        (item.epoch, item.position, item.example_index, token_index)
        for items, mask in zip(plan.step_items, plan.step_masks, strict=True)
        for item, indices in zip(items, mask.selected_token_indices, strict=True)
        for token_index in indices
    ]
    assert len(selected_identities) == 63680
    assert len(set(selected_identities)) == 63680
    assert plan.boundary_split_occurrence_count > 0
    assert plan.selected_candidate_id_coverage == plan.candidate_id_count == 500
    assert plan.occurrence_with_kept_token_count == plan.occurrence_count == 1000
    assert plan.mandatory_coverage_token_count == 1000

    # The last response token is a proxy for final-answer/EOS position. Its
    # retention rate should remain close to the global hash-sampling rate.
    selected_set = set(selected_identities)
    final_kept = sum(
        (epoch, position, example_index, 70) in selected_set
        for epoch, order in enumerate(epoch_orders)
        for position, example_index in enumerate(order)
    )
    final_retention = final_kept / 1000
    overall_retention = 63680 / 70000
    assert abs(final_retention - overall_retention) < 0.02


def test_hash_uniform_mask_is_deterministic_and_not_tail_truncation() -> None:
    example = {
        "input_ids": list(range(101)),
        "attention_mask": [1] * 101,
        "labels": [-100] + list(range(1, 101)),
    }
    items = balanced_optimizer_step_plan(
        epoch_orders=[[0]], optimizer_steps=1
    )[0]
    left = build_hash_uniform_step_mask(
        step_items=items,
        tokenized_examples=[example],
        record_ids=["candidate-a"],
        kept_tokens=50,
        seed=17,
    )
    repeated = build_hash_uniform_step_mask(
        step_items=items,
        tokenized_examples=[example],
        record_ids=["candidate-a"],
        kept_tokens=50,
        seed=17,
    )
    other_seed = build_hash_uniform_step_mask(
        step_items=items,
        tokenized_examples=[example],
        record_ids=["candidate-a"],
        kept_tokens=50,
        seed=29,
    )

    assert left == repeated
    assert left.mask_sha256 != other_seed.mask_sha256
    selected = set(left.selected_token_indices[0])
    assert selected != set(range(1, 51))
    assert any(index > 50 for index in selected)
    assert any(index <= 50 for index in selected)


def test_supervision_token_cap_requires_equal_step_dose() -> None:
    with pytest.raises(ValueError, match="divisible"):
        supervision_tokens_per_step(
            supervision_token_cap=63681,
            optimizer_steps=64,
            policy="hash_uniform_v1",
        )
    with pytest.raises(ValueError, match="unsupported"):
        supervision_tokens_per_step(
            supervision_token_cap=63680,
            optimizer_steps=64,
            policy="tail_clip",
        )


def test_v4_contract_enables_cap_without_controller_cli_flags() -> None:
    assert resolve_token_cap_options(
        cli_supervision_token_cap=None,
        cli_token_cap_policy=None,
        contract_supervision_token_cap=63680,
        contract_token_cap_policy="hash_uniform_v1",
    ) == (63680, "hash_uniform_v1")


def test_v3_contract_preserves_uncapped_default_and_cli_has_priority() -> None:
    assert resolve_token_cap_options(
        cli_supervision_token_cap=None,
        cli_token_cap_policy=None,
        contract_supervision_token_cap=None,
        contract_token_cap_policy=None,
    ) == (None, None)
    assert resolve_token_cap_options(
        cli_supervision_token_cap=64000,
        cli_token_cap_policy="hash_uniform_v1",
        contract_supervision_token_cap=None,
        contract_token_cap_policy=None,
    ) == (64000, "hash_uniform_v1")


def test_v4_contract_rejects_cli_override_but_allows_exact_repetition() -> None:
    assert resolve_token_cap_options(
        cli_supervision_token_cap=63680,
        cli_token_cap_policy="hash_uniform_v1",
        contract_supervision_token_cap=63680,
        contract_token_cap_policy="hash_uniform_v1",
    ) == (63680, "hash_uniform_v1")
    with pytest.raises(ValueError, match="frozen contract"):
        resolve_token_cap_options(
            cli_supervision_token_cap=64000,
            cli_token_cap_policy="hash_uniform_v1",
            contract_supervision_token_cap=63680,
            contract_token_cap_policy="hash_uniform_v1",
        )
    with pytest.raises(ValueError, match="provided together"):
        resolve_token_cap_options(
            cli_supervision_token_cap=63680,
            cli_token_cap_policy=None,
            contract_supervision_token_cap=63680,
            contract_token_cap_policy="hash_uniform_v1",
        )
