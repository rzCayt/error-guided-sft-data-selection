from eg_sft.evaluation.gsm8k_generation import (
    PROMPT_VERSION,
    build_evaluation_prompt,
    score_generation,
)


def _record() -> dict[str, object]:
    return {
        "record_id": "gsm8k-train-0001-abc",
        "source_index": 1,
        "question_sha256": "abc",
    }


def test_prompt_contains_frozen_final_line_contract() -> None:
    prompt = build_evaluation_prompt("What is 2 + 3?")
    assert "Final answer: <number>" in prompt
    assert "What is 2 + 3?" in prompt


def test_score_generation_preserves_wrong_model_number() -> None:
    result = score_generation(
        record=_record(),
        gold_answer_text="2 + 3 = 5.\n#### 5",
        generated_text="I made a mistake.\nFinal answer: 6",
    )
    assert result["prompt_version"] == PROMPT_VERSION
    assert result["parsed_prediction"] == "6"
    assert result["numeric_correct"] is False


def test_parse_failure_counts_as_incorrect() -> None:
    result = score_generation(
        record=_record(),
        gold_answer_text="2 + 3 = 5.\n#### 5",
        generated_text="The answer is five.",
    )
    assert result["parse_status"] == "missing_final_marker"
    assert result["parsed_prediction"] is None
    assert result["numeric_correct"] is False
