import json
from pathlib import Path

import pytest

from eg_sft.experiment.b500_formal_audit import (
    audit_checkpoint_directory,
    audit_formal_output_scope,
    audit_thermal_events,
    audit_training_contract,
    compare_tokenizer_texts,
    read_jsonl,
    write_json_exclusive,
    write_sha256_sidecar_exclusive,
)
from eg_sft.experiment.formal_runtime import file_sha256


class _FakeTokenizer:
    def __init__(self, *, offset: int = 0) -> None:
        self.offset = offset

    def __call__(self, value, **_kwargs):
        def encode(text: str) -> list[int]:
            return [ord(character) + self.offset for character in text]

        if isinstance(value, list):
            return {"input_ids": [encode(text) for text in value]}
        return {"input_ids": encode(value)}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_exclusive_audit_artifact_and_sidecar_never_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    write_json_exclusive(output, {"status": "PASS"})
    sidecar = write_sha256_sidecar_exclusive(output)
    assert sidecar.read_text(encoding="ascii") == f"{file_sha256(output)}  audit.json\n"
    with pytest.raises(FileExistsError):
        write_json_exclusive(output, {"status": "changed"})
    with pytest.raises(FileExistsError):
        write_sha256_sidecar_exclusive(output)


def test_jsonl_reader_rejects_blank_lines_and_missing_final_newline(tmp_path: Path) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text('{"index": 0}\n', encoding="utf-8")
    assert read_jsonl(valid) == [{"index": 0}]

    blank = tmp_path / "blank.jsonl"
    blank.write_text('{"index": 0}\n\n', encoding="utf-8")
    with pytest.raises(ValueError, match="blank JSONL"):
        read_jsonl(blank)

    unterminated = tmp_path / "unterminated.jsonl"
    unterminated.write_text('{"index": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="final newline"):
        read_jsonl(unterminated)


def test_tokenizer_comparison_requires_exact_ids() -> None:
    report = compare_tokenizer_texts(
        reference=_FakeTokenizer(),
        saved=_FakeTokenizer(),
        texts=["abc", "42"],
        tokenizer_kwargs={},
        single_item_batch=False,
    )
    assert report["compared_count"] == 2
    assert report["exact_token_id_equality"] is True

    with pytest.raises(ValueError, match="tokenizer ID mismatch"):
        compare_tokenizer_texts(
            reference=_FakeTokenizer(),
            saved=_FakeTokenizer(offset=1),
            texts=["abc"],
            tokenizer_kwargs={},
            single_item_batch=True,
        )


def test_training_contract_recomputes_steps_tokens_and_reload_gate() -> None:
    selected = [
        {
            "candidate_id": "a",
            "source_index": 1,
            "prompt_sha256": "p1",
            "response_sha256": "r1",
            "total_tokens": 10,
            "supervised_tokens": 3,
        },
        {
            "candidate_id": "b",
            "source_index": 2,
            "prompt_sha256": "p2",
            "response_sha256": "r2",
            "total_tokens": 12,
            "supervised_tokens": 5,
        },
    ]
    metrics = {
        "status": "PASS",
        "strategy": "random",
        "seed": 17,
        "selected_count": 2,
        "epochs": 2,
        "optimizer_steps_planned": 2,
        "optimizer_steps_completed": 2,
        "supervised_tokens_seen": 16,
        "trainable_parameters": 8,
        "adapter_reload_loss_absolute_difference": 0.0,
        "adapter_reload_gate_difference_at_most_1e_6": True,
    }
    report = audit_training_contract(
        metrics=metrics,
        token_audit=selected,
        selected=selected,
        training_config={"epochs": 2, "gradient_accumulation_steps": 2},
        strategy="random",
        seed=17,
        adapter_parameter_count=8,
    )
    assert report["optimizer_steps"] == 2
    assert report["supervised_tokens_seen"] == 16

    metrics["optimizer_steps_completed"] = 1
    with pytest.raises(ValueError, match="optimizer_steps_completed"):
        audit_training_contract(
            metrics=metrics,
            token_audit=selected,
            selected=selected,
            training_config={"epochs": 2, "gradient_accumulation_steps": 2},
            strategy="random",
            seed=17,
            adapter_parameter_count=8,
        )


def test_checkpoint_audit_verifies_all_pairs_hashes_and_progress(tmp_path: Path) -> None:
    binding = {
        "git_commit": "commit",
        "run_config_hash": "config",
        "strategy": "random",
        "seed": 17,
        "selected_id_sha256": "selected",
    }
    progress = [(0, 0), (2, 1), (4, 2), (5, 3)]
    for micro_batch, step in progress:
        stem = f"checkpoint_mb_{micro_batch:04d}_step_{step:03d}_test"
        payload = tmp_path / f"{stem}.pt"
        payload.write_bytes(f"payload-{step}".encode())
        _write_json(
            tmp_path / f"{stem}.json",
            {
                **binding,
                "checkpoint_file": payload.name,
                "checkpoint_sha256": file_sha256(payload),
                "next_micro_batch_index": micro_batch,
                "optimizer_steps": step,
            },
        )
    report = audit_checkpoint_directory(
        checkpoint_directory=tmp_path,
        expected_binding=binding,
        total_micro_batches=5,
        gradient_accumulation_steps=2,
        optimizer_steps=3,
        verify_payload_hashes=True,
    )
    assert report["sidecar_count"] == 4
    assert report["last_progress"] == [5, 3]
    assert report["all_payload_hashes_verified"] is True

    (tmp_path / "checkpoint_mb_0004_step_002_test.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        audit_checkpoint_directory(
            checkpoint_directory=tmp_path,
            expected_binding=binding,
            total_micro_batches=5,
            gradient_accumulation_steps=2,
            optimizer_steps=3,
            verify_payload_hashes=True,
        )


def test_thermal_and_output_scope_gates() -> None:
    report = audit_thermal_events(
        events=[
            {
                "event": "thermal_pause",
                "stage": "training_after",
                "initial_sample": {"temperature_c": 76},
                "resume_sample": {"temperature_c": 62},
            }
        ],
        pause_at_c=75,
        resume_at_c=62,
        hard_stop_at_c=80,
    )
    assert report["max_pause_temperature_c"] == 76
    with pytest.raises(ValueError, match="hard-stop"):
        audit_thermal_events(
            events=[
                {
                    "event": "thermal_pause",
                    "stage": "evaluation_after",
                    "initial_sample": {"temperature_c": 80},
                    "resume_sample": {"temperature_c": 62},
                }
            ],
            pause_at_c=75,
            resume_at_c=62,
            hard_stop_at_c=80,
        )

    scope = audit_formal_output_scope(
        actual_jobs=[("random", 17), ("rds_all", 17)],
        job_order=[
            {"strategy": "random", "seed": 17},
            {"strategy": "rds_all", "seed": 17},
            {"strategy": "rds_error", "seed": 17},
        ],
        current_job=("rds_all", 17),
    )
    assert scope["completed_job_count"] == 2
    with pytest.raises(ValueError, match="scope mismatch"):
        audit_formal_output_scope(
            actual_jobs=[("random", 17), ("rds_error", 17)],
            job_order=[
                {"strategy": "random", "seed": 17},
                {"strategy": "rds_all", "seed": 17},
                {"strategy": "rds_error", "seed": 17},
            ],
            current_job=("rds_error", 17),
        )
