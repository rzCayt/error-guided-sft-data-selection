"""RDS+ representations and deterministic round-robin candidate scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from eg_sft.training.overfit import gsm8k_training_text


RDS_FORMAT_VERSION = "qwen2_5_rds_weighted_mean_v1"


def weighted_mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool token states with linearly increasing weights, then L2-normalize.

    This follows the public RDS+ weighted-mean idea while explicitly excluding
    padding tokens. For an unpadded sequence, token positions receive weights
    1, 2, ..., sequence_length.
    """

    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("hidden states and attention mask shapes do not match")
    if torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError("every sequence must contain at least one attended token")

    positions = torch.arange(
        1,
        last_hidden_state.shape[1] + 1,
        device=last_hidden_state.device,
        dtype=last_hidden_state.dtype,
    ).unsqueeze(0)
    weights = positions * attention_mask.to(last_hidden_state.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    pooled = torch.sum(last_hidden_state * weights.unsqueeze(-1), dim=1)
    return F.normalize(pooled, p=2, dim=1)


def format_tulu_rds_text(
    messages: Sequence[dict[str, str]],
    *,
    eos_token: str,
) -> str:
    """Match the public RDS+ role-tag format for one instruction example."""

    if not messages:
        raise ValueError("messages must be non-empty")
    pieces: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "").strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role}")
        if not content:
            raise ValueError(f"empty {role} message")
        suffix = eos_token if role == "assistant" else ""
        pieces.append(f"<|{role}|>\n{content}{suffix}\n")
    if messages[-1].get("role") != "assistant":
        raise ValueError("candidate must end with an assistant response")
    return "".join(pieces).strip()


def format_gsm8k_rds_text(
    *,
    question: str,
    answer: str,
    eos_token: str,
) -> str:
    """Build the frozen query-plus-gold-response text for GSM8K."""

    _, response = gsm8k_training_text(question, answer)
    prompt = (
        "<|user|>\nAnswer the following question.\n\n"
        f"Question: {question.strip()}\n"
        "<|assistant|>\nAnswer:"
    )
    return f"{prompt}{response}{eos_token}"


@torch.no_grad()
def encode_rds_texts(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Encode texts as CPU float32 unit vectors."""

    if not texts:
        raise ValueError("texts must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        pooled = weighted_mean_pool(outputs.hidden_states[-1], attention_mask)
        chunks.append(pooled.float().cpu())
    return torch.cat(chunks, dim=0)


def cosine_similarity_matrix(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return query-by-candidate cosine similarities."""

    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if query_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError("query and candidate embedding dimensions differ")
    queries = F.normalize(query_embeddings.float(), p=2, dim=1)
    candidates = F.normalize(candidate_embeddings.float(), p=2, dim=1)
    return queries @ candidates.T


def round_robin_order(
    similarity_matrix: torch.Tensor,
    *,
    count: int | None = None,
) -> list[int]:
    """Select each query's best unused candidate in cyclic query order."""

    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be two-dimensional")
    query_count, candidate_count = similarity_matrix.shape
    if query_count == 0 or candidate_count == 0:
        raise ValueError("similarity_matrix must be non-empty")
    selected_count = candidate_count if count is None else count
    if selected_count < 0 or selected_count > candidate_count:
        raise ValueError("count must be between zero and candidate count")

    scores = similarity_matrix.detach().float().cpu()
    used = torch.zeros(candidate_count, dtype=torch.bool)
    order: list[int] = []
    for step in range(selected_count):
        query_index = step % query_count
        row = scores[query_index].clone()
        row[used] = -torch.inf
        candidate_index = int(torch.argmax(row).item())
        used[candidate_index] = True
        order.append(candidate_index)
    return order


def rank_scores(order: Sequence[int], *, candidate_count: int) -> list[float]:
    """Convert a complete selection order to scores in [0, 1]."""

    if len(order) != candidate_count or set(order) != set(range(candidate_count)):
        raise ValueError("order must contain every candidate exactly once")
    if candidate_count == 1:
        return [1.0]
    scores = [0.0] * candidate_count
    denominator = candidate_count - 1
    for rank, candidate_index in enumerate(order):
        scores[candidate_index] = 1.0 - rank / denominator
    return scores
