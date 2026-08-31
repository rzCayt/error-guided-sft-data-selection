import pytest

from eg_sft.training.tulu import tulu_response_only_parts


def test_tulu_response_only_parts_masks_prior_turns_by_construction() -> None:
    prompt, response = tulu_response_only_parts(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
            {"role": "assistant", "content": "Final answer"},
        ],
        eos_token="<eos>",
    )
    assert "First answer<eos>" in prompt
    assert prompt.endswith("<|assistant|>\n")
    assert response == "Final answer"


def test_tulu_response_only_parts_rejects_missing_final_response() -> None:
    with pytest.raises(ValueError, match="end with assistant"):
        tulu_response_only_parts(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Question"},
            ],
            eos_token="<eos>",
        )
