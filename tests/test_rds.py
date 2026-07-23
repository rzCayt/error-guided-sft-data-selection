import torch

from eg_sft.selection.rds import (
    cosine_similarity_matrix,
    format_tulu_rds_text,
    rank_scores,
    round_robin_order,
    weighted_mean_pool,
)


def test_weighted_mean_pool_matches_unpadded_manual_result() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    mask = torch.tensor([[1, 1, 1]])
    pooled = weighted_mean_pool(hidden, mask)
    expected = torch.tensor([[4.0 / 6.0, 5.0 / 6.0]])
    expected = torch.nn.functional.normalize(expected, p=2, dim=1)
    assert torch.allclose(pooled, expected)


def test_weighted_mean_pool_excludes_padding() -> None:
    short = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    padded = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [999.0, 999.0]]])
    short_result = weighted_mean_pool(short, torch.tensor([[1, 1]]))
    padded_result = weighted_mean_pool(padded, torch.tensor([[1, 1, 0]]))
    assert torch.allclose(short_result, padded_result)


def test_tulu_format_requires_final_assistant_and_adds_eos() -> None:
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    text = format_tulu_rds_text(messages, eos_token="<eos>")
    assert text == "<|user|>\nQuestion\n<|assistant|>\nAnswer<eos>"


def test_round_robin_uses_query_group_and_never_repeats_candidates() -> None:
    all_query_similarity = torch.tensor(
        [
            [0.9, 0.8, 0.1],
            [0.1, 0.95, 0.7],
        ]
    )
    error_query_similarity = all_query_similarity[1:]
    all_order = round_robin_order(all_query_similarity)
    error_order = round_robin_order(error_query_similarity)
    assert all_order == [0, 1, 2]
    assert error_order == [1, 2, 0]
    assert len(set(all_order)) == 3
    assert rank_scores(all_order, candidate_count=3) == [1.0, 0.5, 0.0]


def test_cosine_similarity_is_invariant_to_positive_rescaling() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    candidates = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    original = cosine_similarity_matrix(queries, candidates)
    rescaled = cosine_similarity_matrix(queries * 7, candidates * 11)
    assert torch.allclose(original, torch.eye(2))
    assert torch.allclose(original, rescaled)
