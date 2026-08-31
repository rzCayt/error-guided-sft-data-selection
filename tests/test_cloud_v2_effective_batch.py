import torch

from eg_sft.training.effective_batch import (
    build_training_micro_batches,
    normalize_gradients_by_token_count,
    optimizer_steps_for_examples,
    shifted_response_loss_sums,
    should_write_checkpoint,
    validate_micro_batch_contract,
)


def _normalized_gradient(logits: torch.Tensor, labels: torch.Tensor, chunks: list[slice]) -> torch.Tensor:
    candidate = logits.detach().clone().requires_grad_(True)
    total_tokens = 0
    for chunk in chunks:
        loss_sums, token_counts = shifted_response_loss_sums(
            logits=candidate[chunk],
            labels=labels[chunk],
        )
        loss_sums.sum().backward()
        total_tokens += int(token_counts.sum().item())
    normalize_gradients_by_token_count([candidate], response_token_count=total_tokens)
    assert candidate.grad is not None
    return candidate.grad.detach().clone()


def test_response_token_normalization_is_micro_batch_invariant() -> None:
    generator = torch.Generator().manual_seed(20260722)
    logits = torch.randn(4, 6, 9, generator=generator, dtype=torch.float64)
    labels = torch.tensor(
        [
            [-100, -100, 1, 2, 3, 4],
            [-100, -100, -100, 4, 5, -100],
            [-100, 6, 7, 8, -100, -100],
            [-100, -100, -100, -100, 2, 1],
        ],
        dtype=torch.long,
    )
    full = _normalized_gradient(logits, labels, [slice(0, 4)])
    pairs = _normalized_gradient(logits, labels, [slice(0, 2), slice(2, 4)])
    singles = _normalized_gradient(
        logits,
        labels,
        [slice(index, index + 1) for index in range(4)],
    )
    torch.testing.assert_close(pairs, full, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(singles, full, rtol=1e-12, atol=1e-12)


def test_all_four_profiles_keep_effective_batch_sixteen() -> None:
    for micro_batch, accumulation in ((1, 16), (2, 8), (4, 4), (8, 2)):
        validate_micro_batch_contract(
            micro_batch_size=micro_batch,
            gradient_accumulation_steps=accumulation,
            nominal_effective_batch_size=16,
        )


def test_flat_micro_batches_preserve_epoch_boundary_and_resume_suffix() -> None:
    batches = build_training_micro_batches(
        epoch_orders=[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
        micro_batch_size=4,
    )
    assert [len(batch) for batch in batches] == [4, 4, 2]
    flattened = [
        (item.epoch, item.position, item.example_index)
        for batch in batches
        for item in batch
    ]
    assert flattened == [
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
        (0, 3, 3),
        (0, 4, 4),
        (1, 0, 4),
        (1, 1, 3),
        (1, 2, 2),
        (1, 3, 1),
        (1, 4, 0),
    ]
    resumed = batches[2:]
    assert [item.example_index for batch in resumed for item in batch] == [1, 0]


def test_checkpoint_interval_is_ten_and_final_is_always_durable() -> None:
    saved_steps = [
        step
        for step in range(1, 64)
        if should_write_checkpoint(
            optimizer_step=step,
            optimizer_steps_planned=63,
            checkpoint_every_optimizer_steps=10,
        )
    ]
    assert saved_steps == [10, 20, 30, 40, 50, 60, 63]
    assert optimizer_steps_for_examples(
        example_count=1000,
        nominal_effective_batch_size=16,
    ) == 63
