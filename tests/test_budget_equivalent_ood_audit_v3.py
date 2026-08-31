from __future__ import annotations

import pytest

from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (
    canonical_json_bytes,
    masked_metrics,
    write_bytes_exclusive_or_verify,
)


def test_masked_metrics_excludes_accuracy_and_correct_counts() -> None:
    payload = masked_metrics(
        {
            "record_count": 3,
            "unique_record_id_count": 3,
            "parser_rows_recomputed_from_raw_text": 3,
            "ordered_frozen_membership": True,
            "gold_hashes_match": True,
            "metrics": {
                "numeric_correct_count": 2,
                "exact_numeric_accuracy": 2 / 3,
            },
        }
    )
    text = canonical_json_bytes(payload).decode("utf-8")
    assert "accuracy" not in text.replace("accuracy_withheld", "")
    assert "correct_count" not in text
    assert payload["accuracy_withheld"] is True


def test_exclusive_or_verify_is_idempotent_and_fail_closed(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    content = b'{"status":"PASS"}\n'
    write_bytes_exclusive_or_verify(path, content)
    write_bytes_exclusive_or_verify(path, content)
    with pytest.raises(ValueError, match="existing audit artifact differs"):
        write_bytes_exclusive_or_verify(path, b'{"status":"FAIL"}\n')
