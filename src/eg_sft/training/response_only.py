"""Response-only tokenization and padding for causal language-model SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


IGNORE_INDEX = -100


class FastTokenizerLike(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]: ...


def tokenize_response_only(
    tokenizer: FastTokenizerLike,
    *,
    prompt: str,
    response: str,
    max_length: int,
    add_eos: bool = True,
) -> dict[str, list[int]]:
    """Tokenize one example and mask every non-response label.

    A fast tokenizer with ``offset_mapping`` support is required. Tokens whose
    character span overlaps the response are supervised; prompt and special
    tokens receive ``IGNORE_INDEX``. An EOS token is appended and supervised
    when space remains.
    """

    if not prompt:
        raise ValueError("prompt must be non-empty")
    if not response:
        raise ValueError("response must be non-empty")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    full_text = prompt + response
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise ValueError("tokenizer must provide offset_mapping")

    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if not (len(input_ids) == len(attention_mask) == len(offsets)):
        raise ValueError("tokenizer returned inconsistent sequence lengths")

    response_start = len(prompt)
    labels: list[int] = []
    for token_id, attended, (start, end) in zip(
        input_ids, attention_mask, offsets, strict=True
    ):
        is_text_token = end > start
        overlaps_response = end > response_start
        labels.append(
            token_id if attended and is_text_token and overlaps_response else IGNORE_INDEX
        )

    eos_token_id = tokenizer.eos_token_id
    if add_eos and eos_token_id is not None and len(input_ids) < max_length:
        input_ids.append(eos_token_id)
        attention_mask.append(1)
        labels.append(eos_token_id)

    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("response was fully truncated; no supervised tokens remain")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass(frozen=True)
class ResponseOnlyCollator:
    """Right-pad response-only examples into PyTorch tensors."""

    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")

        max_length = max(len(example["input_ids"]) for example in examples)
        batch_input_ids: list[list[int]] = []
        batch_attention: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for example in examples:
            length = len(example["input_ids"])
            if not (
                length
                == len(example["attention_mask"])
                == len(example["labels"])
            ):
                raise ValueError("example fields must have equal lengths")
            padding = max_length - length
            batch_input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            batch_attention.append(example["attention_mask"] + [0] * padding)
            batch_labels.append(
                example["labels"] + [self.label_pad_token_id] * padding
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
