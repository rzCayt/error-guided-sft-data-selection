from __future__ import annotations

import json

import pytest

from eg_sft.experiment.real_tokenizer_dose_audit import (
    EVIDENCE_TYPE,
    build_real_tokenizer_dry_run_report,
    write_json_exclusive,
)


def _real_shape_fixture():
    selected = []
    examples = []
    audits = []
    for index in range(500):
        candidate_id = f"candidate-{index:03d}"
        supervised = 64 + (index % 2)
        total = supervised + 2
        selected.append(
            {
                "candidate_id": candidate_id,
                "total_tokens": total,
                "supervised_tokens": supervised,
            }
        )
        examples.append(
            {
                "input_ids": [1] * total,
                "attention_mask": [1] * total,
                "labels": [-100, -100] + [1] * supervised,
            }
        )
        audits.append(
            {
                "candidate_id": candidate_id,
                "total_tokens": total,
                "supervised_tokens": supervised,
            }
        )
    return selected, examples, audits


def test_build_real_tokenizer_report_passes_exact_v4_gates():
    selected, examples, audits = _real_shape_fixture()
    report = build_real_tokenizer_dry_run_report(
        cell_id="dose_rep1_random_free_mix_train17_cap63680",
        selected=selected,
        tokenized_examples=examples,
        token_audit=audits,
        epochs=2,
        optimizer_steps=64,
        seed=17,
        supervision_token_cap=63680,
        token_cap_policy="hash_uniform_v1",
        bindings={"code": {"relevant_code_bundle_sha256": "a" * 64}},
    )

    assert report["status"] == "PASS"
    assert report["evidence_type"] == EVIDENCE_TYPE
    assert report["dose_plan"]["occurrence_count"] == 1000
    assert report["dose_plan"]["optimizer_steps"] == 64
    assert report["dose_plan"]["tokens_per_optimizer_step"] == 995
    assert report["dose_plan"]["kept_supervision_exposure_tokens"] == 63680
    assert all(report["gates"].values())
    assert {
        row["kept_response_supervision_tokens"]
        for row in report["dose_plan"]["steps"]
    } == {995}
    assert {
        row["evidence_type"] for row in report["prior_metadata_artifacts"]
    } == {"metadata_simulation"}


def test_real_tokenizer_report_rejects_formal_token_audit_drift():
    selected, examples, audits = _real_shape_fixture()
    audits[0]["supervised_tokens"] += 1
    with pytest.raises(ValueError, match="formal token audit differs"):
        build_real_tokenizer_dry_run_report(
            cell_id="dose_rep1_random_free_mix_train17_cap63680",
            selected=selected,
            tokenized_examples=examples,
            token_audit=audits,
            epochs=2,
            optimizer_steps=64,
            seed=17,
            supervision_token_cap=63680,
            token_cap_policy="hash_uniform_v1",
            bindings={},
        )


def test_write_json_exclusive_refuses_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    write_json_exclusive(path, {"status": "PASS"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "PASS"}
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"status": "SECOND"})
