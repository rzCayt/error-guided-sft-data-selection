from __future__ import annotations

import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_budget_equivalent_ood_eval_worker as ood_worker  # noqa: E402
import run_identifiable_budget_v4_worker as v4_worker  # noqa: E402

from eg_sft.evaluation.identifiable_batch_backend import (  # noqa: E402
    QUALIFICATION_GATES,
)


def test_v4_controller_accepts_the_audit_gate_schema(tmp_path: Path) -> None:
    report = tmp_path / "qualification.json"
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gates": {name: True for name in QUALIFICATION_GATES},
            }
        ),
        encoding="utf-8",
    )
    assert v4_worker._qualification(report)["status"] == "PASS"

    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["gates"][
        "token_ids_equal_or_full_shadow_semantic_equivalence"
    ] = False
    report.write_text(json.dumps(payload), encoding="utf-8")
    try:
        v4_worker._qualification(report)
    except ValueError as error:
        assert "gates are incomplete" in str(error)
    else:
        raise AssertionError("controller accepted failed token equivalence")


def test_ood_batch_validates_every_source_row(monkeypatch) -> None:
    records = [
        {"record_id": "a", "source_index": 0},
        {"record_id": "b", "source_index": 1},
        {"record_id": "c", "source_index": 2},
    ]
    source = [
        {"question": "q0", "answer": "0"},
        {"question": "q1", "answer": "1"},
        {"question": "q2", "answer": "2"},
    ]
    validated = []

    def fake_validate(*, record, raw_row, answer_field):
        validated.append((record["record_id"], raw_row["question"], answer_field))
        return raw_row["answer"]

    monkeypatch.setattr(ood_worker, "validate_source_row", fake_validate)
    monkeypatch.setattr(
        ood_worker,
        "build_ood_prompt",
        lambda dataset, row: f"{dataset}:{row['question']}",
    )
    gold, prompts = ood_worker._validated_ood_batch(
        records=records,
        source=source,
        source_spec={"answer_field": "answer"},
        dataset="svamp",
    )
    assert validated == [
        ("a", "q0", "answer"),
        ("b", "q1", "answer"),
        ("c", "q2", "answer"),
    ]
    assert gold == ["0", "1", "2"]
    assert prompts == ["svamp:q0", "svamp:q1", "svamp:q2"]
