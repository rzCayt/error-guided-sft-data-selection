from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_identifiable_base_reference import TOTAL_RECORDS, _batch_outputs  # noqa: E402


def test_base_reference_total_is_four_frozen_datasets() -> None:
    assert TOTAL_RECORDS == 3841


def test_batch_outputs_rejects_more_prompts_than_declared_batch() -> None:
    class Encoded(dict):
        def to(self, _device):
            return self

    class Tensor:
        shape = (2, 3)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, *_args, **_kwargs):
            return Encoded(input_ids=Tensor())

        def decode(self, row, skip_special_tokens=True):
            return " ".join(map(str, row))

    class Generated:
        def tolist(self):
            return [[1, 2, 3, 4], [1, 2, 3, 5]]

    class Model:
        def generate(self, **_kwargs):
            return Generated()

    with pytest.raises(ValueError, match="batch cardinality"):
        _batch_outputs(
            model=Model(),
            tokenizer=Tokenizer(),
            prompts=["a", "b"],
            batch_size=1,
            max_input_length=10,
            max_new_tokens=2,
            device="cpu",
        )
