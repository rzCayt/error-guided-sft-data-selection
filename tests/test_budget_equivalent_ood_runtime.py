from __future__ import annotations

from eg_sft.data.public_gsm8k import sha256_text
from eg_sft.evaluation.arithmetic_ood import PROMPT_VERSION, build_ood_record
from eg_sft.experiment.budget_equivalent_ood_runtime import (
    audit_complete_dataset,
    canonical_source_row_sha256,
    contiguous_shard,
    recompute_ood_metrics,
    validate_resume_worker_manifest,
    validate_source_row,
    validate_worker_prefix,
)


def _frozen_record() -> tuple[dict[str, object], dict[str, str]]:
    raw = {
        "body": "Mia has four apples.",
        "question": "How many apples does Mia have?",
        "answer": "4 (apples)",
    }
    built = build_ood_record(
        dataset="asdiv_numeric",
        source_index=7,
        row=raw,
        answer_field="answer",
    )
    gold = str(built.pop("gold_value"))
    built.pop("numeric_eligible")
    built["source_row_sha256"] = canonical_source_row_sha256(raw)
    built["gold_value_sha256"] = sha256_text(gold)
    return built, raw


def _output(record: dict[str, object], *, correct: bool = True) -> dict[str, object]:
    return {
        "record_id": record["record_id"],
        "dataset": record["dataset"],
        "source_index": record["source_index"],
        "question_sha256": record["question_sha256"],
        "prompt_version": PROMPT_VERSION,
        "strict_parse_status": "ok",
        "parse_mode": "strict_final_marker",
        "gold_value": "4",
        "numeric_correct": correct,
    }


def test_validate_source_row_recovers_gold_from_frozen_hashes() -> None:
    record, raw = _frozen_record()
    assert validate_source_row(record=record, raw_row=raw, answer_field="answer") == "4"


def test_validate_source_row_rejects_source_mutation() -> None:
    record, raw = _frozen_record()
    raw["answer"] = "5 (apples)"
    try:
        validate_source_row(record=record, raw_row=raw, answer_field="answer")
    except ValueError as error:
        assert "source row hash changed" in str(error)
    else:
        raise AssertionError("mutated source row was accepted")


def test_contiguous_shards_cover_each_record_once() -> None:
    records = [{"record_id": str(index)} for index in range(10)]
    shards = [contiguous_shard(records, shard_index=index, shard_count=3) for index in range(3)]
    assert [(start, end) for start, end, _ in shards] == [(0, 3), (3, 6), (6, 10)]
    assert [row["record_id"] for _, _, shard in shards for row in shard] == [
        str(index) for index in range(10)
    ]


def test_prefix_and_complete_audit_enforce_order_and_gold_hash() -> None:
    record, _ = _frozen_record()
    row = _output(record)
    assert validate_worker_prefix(rows=[row], frozen_records=[record]) == 1
    report = audit_complete_dataset(rows=[row], frozen_records=[record])
    assert report["status"] == "PASS"
    assert report["metrics"]["exact_numeric_accuracy"] == 1.0


def test_recompute_metrics_distinguishes_strict_parse_from_correctness() -> None:
    record, _ = _frozen_record()
    correct = _output(record, correct=True)
    wrong = _output(record, correct=False)
    wrong["record_id"] = "different"
    metrics = recompute_ood_metrics([correct, wrong])
    assert metrics["exact_numeric_accuracy"] == 0.5
    assert metrics["strict_parse_rate"] == 1.0


def test_resume_manifest_allows_only_gpu_uuid_change() -> None:
    existing = {"cell_id": "c1", "gpu_uuid": "gpu-old", "worker": {"shard": 0}}
    expected = {"cell_id": "c1", "gpu_uuid": "gpu-new", "worker": {"shard": 0}}
    assert validate_resume_worker_manifest(existing=existing, expected=expected) is True
    assert validate_resume_worker_manifest(existing=existing, expected=existing) is False
    changed = dict(expected) | {"cell_id": "c2"}
    try:
        validate_resume_worker_manifest(existing=existing, expected=changed)
    except ValueError as error:
        assert "beyond GPU UUID" in str(error)
    else:
        raise AssertionError("non-GPU manifest mutation was accepted")
