import copy

import pytest

from eg_sft.selection.query_groups import freeze_query_groups


def _fixtures() -> tuple[list[dict], list[dict]]:
    records = [
        {
            "record_id": f"id-{index}",
            "source_index": index,
            "protocol_split": "selection_diagnostic",
            "question_sha256": f"hash-{index}",
        }
        for index in range(6)
    ]
    records.append(
        {
            "record_id": "development-id",
            "source_index": 99,
            "protocol_split": "development",
            "question_sha256": "development-hash",
        }
    )
    outputs = [
        {
            "record_id": f"id-{index}",
            "question_sha256": f"hash-{index}",
            "parse_status": "ok",
            "numeric_correct": index < 3,
        }
        for index in range(6)
    ]
    return records, outputs


def test_freeze_query_groups_is_order_invariant_and_excludes_other_splits() -> None:
    records, outputs = _fixtures()
    first = freeze_query_groups(
        split_records=records,
        diagnostic_outputs=outputs,
        minimum_group_size=3,
    )
    second = freeze_query_groups(
        split_records=list(reversed(records)),
        diagnostic_outputs=list(reversed(outputs)),
        minimum_group_size=3,
    )

    assert first == second
    all_queries, error_queries, manifest = first
    assert len(all_queries) == 6
    assert len(error_queries) == 3
    assert "development-id" not in {row["record_id"] for row in all_queries}
    assert manifest["group_size_gate_passed"] is True
    assert manifest["fallback_required"] is False


def test_freeze_query_groups_rejects_missing_or_unexpected_rows() -> None:
    records, outputs = _fixtures()
    with pytest.raises(ValueError, match="coverage mismatch"):
        freeze_query_groups(
            split_records=records,
            diagnostic_outputs=outputs[:-1],
        )

    unexpected = copy.deepcopy(outputs)
    unexpected[-1]["record_id"] = "not-in-split"
    with pytest.raises(ValueError, match="coverage mismatch"):
        freeze_query_groups(
            split_records=records,
            diagnostic_outputs=unexpected,
        )


def test_freeze_query_groups_rejects_hash_or_label_corruption() -> None:
    records, outputs = _fixtures()
    bad_hash = copy.deepcopy(outputs)
    bad_hash[0]["question_sha256"] = "wrong"
    with pytest.raises(ValueError, match="question hash mismatch"):
        freeze_query_groups(
            split_records=records,
            diagnostic_outputs=bad_hash,
        )

    bad_label = copy.deepcopy(outputs)
    bad_label[0]["numeric_correct"] = 1
    with pytest.raises(ValueError, match="must be boolean"):
        freeze_query_groups(
            split_records=records,
            diagnostic_outputs=bad_label,
        )


def test_freeze_query_groups_reports_failed_size_gate_without_relabeling() -> None:
    records, outputs = _fixtures()
    _, error_queries, manifest = freeze_query_groups(
        split_records=records,
        diagnostic_outputs=outputs,
        minimum_group_size=4,
    )

    assert len(error_queries) == 3
    assert manifest["group_size_gate_passed"] is False
    assert manifest["fallback_required"] is True
