import pytest

from eg_sft.selection.h1a_sample import (
    select_until_eligible_count,
    stratified_candidate_sample,
)


def _candidates() -> list[dict]:
    return [
        {
            "candidate_id": f"{source}-{index}",
            "source_dataset": source,
            "source_index": index,
        }
        for source in ("a", "b", "c")
        for index in range(5)
    ]


def test_stratified_sample_is_input_order_invariant_and_source_balanced() -> None:
    candidates = _candidates()
    first = stratified_candidate_sample(candidates, count=9, seed=20260722)
    second = stratified_candidate_sample(
        list(reversed(candidates)),
        count=9,
        seed=20260722,
    )
    assert first == second
    counts = {
        source: sum(row["source_dataset"] == source for row in first)
        for source in ("a", "b", "c")
    }
    assert counts == {"a": 3, "b": 3, "c": 3}


def test_stratified_sample_rejects_duplicate_ids() -> None:
    candidates = _candidates()
    candidates[-1]["candidate_id"] = candidates[0]["candidate_id"]
    with pytest.raises(ValueError, match="unique"):
        stratified_candidate_sample(candidates, count=3, seed=1)


def test_select_until_eligible_count_supplements_without_reordering() -> None:
    candidates = [{"candidate_id": str(index), "valid": index % 3 != 0} for index in range(10)]
    selected, excluded = select_until_eligible_count(
        candidates,
        count=5,
        is_eligible=lambda row: row["valid"],
    )
    assert [row["candidate_id"] for row in selected] == ["1", "2", "4", "5", "7"]
    assert [row["candidate_id"] for row in excluded] == ["0", "3", "6"]
