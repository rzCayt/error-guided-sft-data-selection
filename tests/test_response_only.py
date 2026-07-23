from typing import Any

import pytest
import torch

from eg_sft.training.response_only import (
    IGNORE_INDEX,
    ResponseOnlyCollator,
    tokenize_response_only,
)


class CharacterTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
        max_length = int(kwargs["max_length"])
        characters = list(text)[:max_length]
        return {
            "input_ids": [ord(character) % 50 + 1 for character in characters],
            "attention_mask": [1] * len(characters),
            "offset_mapping": [(index, index + 1) for index in range(len(characters))],
        }


def test_only_response_and_eos_are_supervised() -> None:
    tokenizer = CharacterTokenizer()
    example = tokenize_response_only(
        tokenizer,
        prompt="Q:",
        response="42",
        max_length=8,
    )

    assert example["labels"][:2] == [IGNORE_INDEX, IGNORE_INDEX]
    assert example["labels"][2:4] == example["input_ids"][2:4]
    assert example["labels"][-1] == tokenizer.eos_token_id
    assert len(example["input_ids"]) == 5


def test_fully_truncated_response_is_rejected() -> None:
    with pytest.raises(ValueError, match="fully truncated"):
        tokenize_response_only(
            CharacterTokenizer(),
            prompt="long prompt",
            response="answer",
            max_length=3,
            add_eos=False,
        )


def test_collator_masks_padding() -> None:
    collator = ResponseOnlyCollator(pad_token_id=0)
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [IGNORE_INDEX, 2, 3],
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [IGNORE_INDEX, 5],
            },
        ]
    )

    assert batch["input_ids"].shape == torch.Size([2, 3])
    assert batch["input_ids"][1].tolist() == [4, 5, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"][1].tolist() == [IGNORE_INDEX, 5, IGNORE_INDEX]
