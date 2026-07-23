import pytest

from eg_sft.training.b500 import (
    selected_id_sha256,
    validate_selection_manifest,
)


def _manifest() -> dict:
    selected = [
        {"candidate_id": f"candidate-{index:03d}"} for index in range(3)
    ]
    return {
        "strategy": "random",
        "budget": 3,
        "selection_seed": 20260722,
        "selected_id_sha256": selected_id_sha256(selected),
        "selected_candidates": selected,
    }


def test_selection_manifest_requires_strategy_budget_seed_and_unique_ids() -> None:
    selected = validate_selection_manifest(
        _manifest(),
        expected_strategy="random",
        expected_budget=3,
        expected_selection_seed=20260722,
    )
    assert [row["candidate_id"] for row in selected] == [
        "candidate-000",
        "candidate-001",
        "candidate-002",
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strategy", "rds_all", "strategy"),
        ("budget", 2, "budget"),
        ("selection_seed", 17, "selection seed"),
    ],
)
def test_selection_manifest_rejects_frozen_field_changes(
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(ValueError, match=message):
        validate_selection_manifest(
            manifest,
            expected_strategy="random",
            expected_budget=3,
            expected_selection_seed=20260722,
        )


def test_selection_manifest_rejects_duplicate_ids() -> None:
    manifest = _manifest()
    manifest["selected_candidates"][2]["candidate_id"] = "candidate-001"
    with pytest.raises(ValueError, match="unique"):
        validate_selection_manifest(
            manifest,
            expected_strategy="random",
            expected_budget=3,
            expected_selection_seed=20260722,
        )
