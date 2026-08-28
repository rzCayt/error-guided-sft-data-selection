from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eg_sft.evaluation.identifiable_backend_qualification_v2 import (
    BATCH_SIZES,
    MODEL_IDS,
    VALIDATED_CONFIG_SHA_KEY,
    aggregate_attempts,
    audit_final,
    audit_smoke128,
    canonical_json_bytes,
    compare_rows,
    file_sha256,
    finalize_run,
    performance_gates,
    record_attempt_finish,
    record_attempt_start,
    run_dir,
    select_smoke_best,
    validate_adapter_root,
    validate_config,
    validate_prefix,
    validate_stop_after_records_request,
    write_exclusive_or_verify,
    write_failure_artifact,
    write_replay_probe,
)
from eg_sft.evaluation.identifiable_batch_backend import (
    validate_qualification_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "identifiable_backend_qualification_v2.json"


def _row(index: int, *, token_offset: int = 0, prediction: str | None = None) -> dict:
    return {
        "record_id": f"r{index:04d}",
        "parsed_prediction": str(index) if prediction is None else prediction,
        "numeric_correct": True,
        "strict_parse_status": "ok",
        "parse_mode": "strict_final_marker",
        "parse_status": "ok",
        "generated_token_ids": [index + token_offset],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _minimal_manifest(
    config: dict,
    *,
    stage: str,
    model_id: str,
    batch_size: int,
    record_count: int,
    rows_sha: str,
) -> dict:
    payload = {
        "stage": stage,
        "model_id": model_id,
        "batch_size": batch_size,
        "record_count": record_count,
        "base_model_repo_id": config["base_model"]["repo_id"],
        "base_model_revision": config["base_model"]["revision"],
        "tokenizer_repo_id": config["tokenizer"]["repo_id"],
        "tokenizer_revision": config["tokenizer"]["revision"],
        "formal_audit_sha256": None,
        "training_metrics_sha256": None,
        "adapter_model_sha256": None,
        "config_sha256": config[VALIDATED_CONFIG_SHA_KEY],
        "matrix_sha256": config["matrix"]["sha256"],
        "script_sha256": "s" * 64,
        "module_sha256": "m" * 64,
        "rows_sha256": rows_sha,
        "peak_memory_bytes": 1,
        "examples_per_second": 1.0,
        "generation_seconds": 1.0,
        "full_wall_seconds": 1.0,
        "accuracy_withheld": True,
    }
    if model_id != "base":
        binding = next(row for row in config["adapters"] if row["adapter_id"] == model_id)
        for field in (
            "formal_audit_sha256",
            "training_metrics_sha256",
            "adapter_model_sha256",
        ):
            payload[field] = binding[field]
    return payload


def _completed(
    root: Path,
    config: dict,
    *,
    stage: str,
    model_id: str,
    batch_size: int,
    count: int,
    speed: float,
    wall: float,
    token_offset: int = 0,
    resume: bool = False,
    attempt_count: int = 1,
    generated_records: int | None = None,
    failed_attempt_count: int = 0,
    replay: bool = True,
) -> Path:
    directory = run_dir(
        root=root, stage=stage, model_id=model_id, batch_size=batch_size
    )
    rows = [_row(index, token_offset=token_offset) for index in range(count)]
    _write_rows(directory / "raw_outputs.jsonl", rows)
    rows_sha = file_sha256(directory / "raw_outputs.jsonl")
    _write_json(
        directory / "metrics.json",
        {
            "status": "PASS",
            "record_count": count,
            "rows_sha256": rows_sha,
            "examples_per_second": speed,
            "full_wall_seconds": wall,
            "resume_observed": resume,
            "attempt_count": attempt_count,
            "generated_records": count if generated_records is None else generated_records,
            "failed_attempt_count": failed_attempt_count,
            "accuracy_withheld": True,
        },
    )
    _write_json(
        directory / "manifest.json",
        _minimal_manifest(
            config,
            stage=stage,
            model_id=model_id,
            batch_size=batch_size,
            record_count=count,
            rows_sha=rows_sha,
        ),
    )
    if replay:
        write_replay_probe(directory)
    return directory


def test_config_binds_current_matrix_and_frozen_schedule() -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    assert file_sha256(ROOT / config["matrix"]["path"]) == config["matrix"]["sha256"]
    assert config["stages"]["smoke128"]["batch_sizes"] == list(BATCH_SIZES)
    assert [row["adapter_id"] for row in config["adapters"]] == list(MODEL_IDS[1:])
    assert config["resume_probe"] == {
        "stage": "smoke128",
        "model_id": "base",
        "batch_size": 2,
        "stop_after_records": 64,
    }


def test_stop_after_records_is_reserved_for_fixed_smoke_resume_probe() -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    validate_stop_after_records_request(
        stage="smoke128",
        model_id="base",
        batch_size=2,
        stop_after_records=64,
        config=config,
    )
    validate_stop_after_records_request(
        stage="confirm512",
        model_id="base",
        batch_size=4,
        stop_after_records=None,
        config=config,
    )
    with pytest.raises(ValueError, match="reserved"):
        validate_stop_after_records_request(
            stage="confirm512",
            model_id="base",
            batch_size=2,
            stop_after_records=64,
            config=config,
        )
    with pytest.raises(ValueError, match="value 64"):
        validate_stop_after_records_request(
            stage="smoke128",
            model_id="base",
            batch_size=2,
            stop_after_records=63,
            config=config,
        )


def test_adapter_requires_formal_training_and_tensor_sha_agreement(tmp_path: Path) -> None:
    root = tmp_path / "adapter"
    tensor = root / "training_complete" / "adapter" / "adapter_model.safetensors"
    tensor.parent.mkdir(parents=True)
    tensor.write_bytes(b"adapter")
    adapter_sha = file_sha256(tensor)
    formal = {
        "status": "PASS",
        "accuracy_withheld": True,
        "artifact_hashes": {"adapter_model": adapter_sha},
    }
    training = {"status": "PASS", "adapter_model_sha256": adapter_sha}
    _write_json(root / "audit" / "formal_cell_audit.json", formal)
    _write_json(root / "training_complete" / "training_metrics.json", training)
    binding = {
        "formal_audit_sha256": file_sha256(root / "audit" / "formal_cell_audit.json"),
        "training_metrics_sha256": file_sha256(
            root / "training_complete" / "training_metrics.json"
        ),
        "adapter_model_sha256": adapter_sha,
    }
    assert validate_adapter_root(adapter_root=root, binding=binding) == binding
    training["adapter_model_sha256"] = "changed"
    _write_json(root / "training_complete" / "training_metrics.json", training)
    binding["training_metrics_sha256"] = file_sha256(
        root / "training_complete" / "training_metrics.json"
    )
    with pytest.raises(ValueError, match="binding disagrees"):
        validate_adapter_root(adapter_root=root, binding=binding)


def test_prefix_rejects_gap_and_duplicate() -> None:
    assert validate_prefix(rows=[{"record_id": "a"}], frozen_ids=["a", "b"]) == 1
    with pytest.raises(ValueError, match="ordered prefix"):
        validate_prefix(rows=[{"record_id": "b"}], frozen_ids=["a", "b"])
    with pytest.raises(ValueError, match="ordered prefix|duplicate"):
        validate_prefix(
            rows=[{"record_id": "a"}, {"record_id": "a"}],
            frozen_ids=["a", "b"],
        )


def test_immutable_write_and_failure_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    assert write_exclusive_or_verify(path, b"same") == "CREATED"
    assert write_exclusive_or_verify(path, b"same") == "VERIFIED_IDENTICAL"
    with pytest.raises(FileExistsError, match="changed"):
        write_exclusive_or_verify(path, b"different")
    failure = write_failure_artifact(
        root=tmp_path,
        stage="smoke128",
        model_id="base",
        batch_size=4,
        error=RuntimeError("out of memory"),
    )
    assert json.loads(failure.read_text(encoding="utf-8"))["status"] == "INELIGIBLE"


def test_performance_thresholds_are_inclusive_and_frozen() -> None:
    assert performance_gates(speedups=[1.5, 1.6], shadow_wall_reduction=0.25) == {
        "throughput": True,
        "shadow_wall": True,
    }
    assert performance_gates(speedups=[1.499], shadow_wall_reduction=0.249) == {
        "throughput": False,
        "shadow_wall": False,
    }


@pytest.mark.parametrize("count", [128, 512])
def test_token_mismatch_fails_at_smoke_and_confirm(count: int) -> None:
    reference = [_row(index) for index in range(count)]
    candidate = [_row(index, token_offset=1) for index in range(count)]
    assert compare_rows(reference=reference, candidate=candidate, record_count=count)[
        "status"
    ] == "FAIL"


def test_shadow_token_fallback_requires_all_six_semantics() -> None:
    reference = [_row(index) for index in range(3841)]
    candidate = [_row(index, token_offset=1) for index in range(3841)]
    report = compare_rows(reference=reference, candidate=candidate, record_count=3841)
    assert report["status"] == "PASS"
    assert report["semantic_fallback_used"] is True
    candidate[10]["parse_mode"] = "changed"
    assert compare_rows(reference=reference, candidate=candidate, record_count=3841)[
        "status"
    ] == "FAIL"


def test_smoke_selects_unique_fastest_mean_and_marks_missing_ineligible(
    tmp_path: Path,
) -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    for model_index, model_id in enumerate(MODEL_IDS):
        _completed(
            tmp_path,
            config,
            stage="smoke128",
            model_id=model_id,
            batch_size=1,
            count=128,
            speed=1.0,
            wall=100.0,
        )
        # Batch2 is absent for one model and must become ineligible, not fatal.
        if model_id != MODEL_IDS[-1]:
            _completed(
                tmp_path,
                config,
                stage="smoke128",
                model_id=model_id,
                batch_size=2,
                count=128,
                speed=1.4,
                wall=80.0,
            )
        for batch_size, speed in ((4, 1.7 + model_index * 0.01), (8, 2.0 + model_index * 0.01)):
            _completed(
                tmp_path,
                config,
                stage="smoke128",
                model_id=model_id,
                batch_size=batch_size,
                count=128,
                speed=speed,
                wall=70.0,
            )
    selection = select_smoke_best(root=tmp_path, config=config)
    assert selection["status"] == "PASS"
    assert selection["best_batch_size"] == 8
    assert next(row for row in selection["candidates"] if row["batch_size"] == 2)[
        "eligible"
    ] is False


def test_actual_stop_resume_sums_all_attempt_times_and_replay(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    frozen = ["a", "b", "c", "d"]
    _write_rows(directory / "raw_outputs.jsonl", [{"record_id": "a"}, {"record_id": "b"}])
    first = record_attempt_start(
        run_directory=directory, start_index=0, stop_after_records=2
    )
    record_attempt_finish(
        run_directory=directory,
        attempt_id=first,
        start_index=0,
        end_index=2,
        generation_seconds=2.0,
        full_wall_seconds=4.0,
        generated_tokens=20,
        peak_memory_bytes=10,
        stopped_early=True,
    )
    with (directory / "raw_outputs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_id": "c"}) + "\n")
        handle.write(json.dumps({"record_id": "d"}) + "\n")
    second = record_attempt_start(
        run_directory=directory, start_index=2, stop_after_records=None
    )
    record_attempt_finish(
        run_directory=directory,
        attempt_id=second,
        start_index=2,
        end_index=4,
        generation_seconds=3.0,
        full_wall_seconds=6.0,
        generated_tokens=30,
        peak_memory_bytes=20,
        stopped_early=False,
    )
    aggregate = aggregate_attempts(directory)
    assert aggregate["resume_observed"] is True
    assert aggregate["generation_seconds"] == 5.0
    assert aggregate["full_wall_seconds"] == 10.0
    manifest = {
        "base_model_repo_id": "repo",
        "base_model_revision": "rev",
        "tokenizer_repo_id": "repo",
        "tokenizer_revision": "rev",
        "formal_audit_sha256": None,
        "training_metrics_sha256": None,
        "adapter_model_sha256": None,
        "config_sha256": "c",
        "matrix_sha256": "m",
        "script_sha256": "s",
        "module_sha256": "x",
    }
    metrics = finalize_run(
        run_directory=directory, frozen_ids=frozen, manifest=manifest
    )
    assert metrics["generation_seconds"] == 5.0
    assert metrics["full_wall_seconds"] == 10.0
    probe = write_replay_probe(directory)
    assert json.loads(probe.read_text(encoding="utf-8"))["bytes_verified_identical"] is True


def _populate_smoke128(
    root: Path,
    config: dict,
    *,
    slow_selected_model: str | None = None,
    missing_replay: tuple[str, int] | None = None,
) -> dict:
    for model_index, model_id in enumerate(MODEL_IDS):
        for batch_size, speed in ((1, 1.0), (2, 1.6), (4, 1.8), (8, 2.2)):
            if batch_size == 8 and slow_selected_model == model_id:
                speed = 1.4
            elif batch_size == 8 and slow_selected_model is not None:
                speed = 2.6 + model_index * 0.01
            is_probe = model_id == "base" and batch_size == 2
            _completed(
                root,
                config,
                stage="smoke128",
                model_id=model_id,
                batch_size=batch_size,
                count=128,
                speed=speed,
                wall=100.0 / speed,
                resume=is_probe,
                attempt_count=2 if is_probe else 1,
                replay=missing_replay != (model_id, batch_size),
            )
    return select_smoke_best(root=root, config=config)


def test_smoke128_audit_closes_only_128_gates(tmp_path: Path) -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    selection = _populate_smoke128(tmp_path, config)
    assert selection["best_batch_size"] == 8

    report = audit_smoke128(root=tmp_path, config=config)

    assert report["status"] == "PASS"
    assert report["audit_gpu_accessed"] is False
    assert report["formal_backend_authorized"] is False
    assert report["larger_stage_artifacts_read"] is False
    assert report["completed_run_count"] == 12
    assert report["qualification_config_sha256"] == config[VALIDATED_CONFIG_SHA_KEY]
    assert report["matrix_sha256"] == config["matrix"]["sha256"]
    assert len(report["audit_module_sha256"]) == 64
    assert all(report["gates"].values())
    smoke_path = tmp_path / "qualification_smoke128.json"
    assert smoke_path.is_file()
    with pytest.raises(ValueError, match="gates are incomplete"):
        validate_qualification_artifact(
            report_path=smoke_path,
            expected_sha256=file_sha256(smoke_path),
        )
    assert not (tmp_path / "qualification_final.json").exists()


def test_smoke128_audit_fails_when_one_selected_speedup_is_below_gate(
    tmp_path: Path,
) -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    selection = _populate_smoke128(
        tmp_path, config, slow_selected_model=MODEL_IDS[-1]
    )
    assert selection["best_batch_size"] == 8

    report = audit_smoke128(root=tmp_path, config=config)

    assert report["status"] == "FAIL"
    assert (
        report["gates"]["selected_batch_speedup_at_least_1_5x_for_each_model"]
        is False
    )
    assert report["formal_backend_authorized"] is False


def test_smoke128_audit_fails_without_byte_bound_replay(tmp_path: Path) -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    _populate_smoke128(
        tmp_path,
        config,
        missing_replay=(MODEL_IDS[1], 4),
    )

    report = audit_smoke128(root=tmp_path, config=config)

    assert report["status"] == "FAIL"
    assert report["gates"]["all_completed_outputs_replay_verified"] is False
    assert "smoke128__random_common_rep1_seed17__b4" in report["replay_failures"]


def test_smoke128_audit_recomputes_qualification_config_sha(
    tmp_path: Path,
) -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    _populate_smoke128(tmp_path, config)
    directory = run_dir(
        root=tmp_path,
        stage="smoke128",
        model_id=MODEL_IDS[1],
        batch_size=8,
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["config_sha256"] = "0" * 64
    _write_json(directory / "manifest.json", manifest)

    report = audit_smoke128(root=tmp_path, config=config)

    assert report["status"] == "FAIL"
    assert report["gates"]["all_completed_rows_and_manifests_bound"] is False
    assert any(
        "qualification config SHA changed" in row["error"]
        for row in report["provenance_errors"]
    )


def test_cli_audit_smoke128_is_cpu_only(tmp_path: Path) -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    _populate_smoke128(tmp_path, config)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_identifiable_backend_qualification_v2.py"),
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(tmp_path),
            "--audit-smoke128",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload == {
        "audit_gpu_accessed": False,
        "formal_backend_authorized": False,
        "status": "PASS",
    }


def test_cli_rejects_multiple_control_modes(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_identifiable_backend_qualification_v2.py"),
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(tmp_path),
            "--audit-smoke128",
            "--audit-final",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "mutually exclusive" in completed.stderr


def test_final_audit_emits_controller_compatible_gates(tmp_path: Path) -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    _write_json(
        tmp_path / "smoke_selection.json",
        {"status": "PASS", "best_batch_size": 4},
    )
    _completed(
        tmp_path,
        config,
        stage="smoke128",
        model_id="base",
        batch_size=2,
        count=128,
        speed=1.5,
        wall=90.0,
        resume=True,
        attempt_count=2,
        generated_records=128,
    )
    for model_id in MODEL_IDS:
        for batch_size, speed, wall in ((1, 1.0, 100.0), (4, 1.6, 70.0)):
            _completed(
                tmp_path,
                config,
                stage="confirm512",
                model_id=model_id,
                batch_size=batch_size,
                count=512,
                speed=speed,
                wall=wall,
                resume=False,
                attempt_count=1,
            )
    shadow = MODEL_IDS[1]
    _completed(
        tmp_path,
        config,
        stage="shadow3841",
        model_id=shadow,
        batch_size=1,
        count=3841,
        speed=1.0,
        wall=100.0,
    )
    _completed(
        tmp_path,
        config,
        stage="shadow3841",
        model_id=shadow,
        batch_size=4,
        count=3841,
        speed=1.6,
        wall=70.0,
        token_offset=1,
    )
    report = audit_final(root=tmp_path, config=config, shadow_adapter_id=shadow)
    assert report["status"] == "PASS"
    assert report["gpu_accessed"] is True
    assert report["method_effectiveness_claimed"] is False
    assert set(report["gates"]) == {
        "row_level_equivalence",
        "token_ids_equal_or_full_shadow_semantic_equivalence",
        "throughput_at_least_1_5x",
        "full_cell_wall_time_reduction_at_least_25_percent",
        "resume_without_gap_or_duplicate",
        "output_non_overwrite",
    }
    final_path = tmp_path / "qualification_final.json"
    assert validate_qualification_artifact(
        report_path=final_path,
        expected_sha256=file_sha256(final_path),
    )["status"] == "PASS"


def test_final_audit_fails_when_confirm_performance_run_was_resumed(
    tmp_path: Path,
) -> None:
    config = validate_config(repo_root=ROOT, config_path=CONFIG_PATH)
    _write_json(
        tmp_path / "smoke_selection.json",
        {"status": "PASS", "best_batch_size": 4},
    )
    _completed(
        tmp_path,
        config,
        stage="smoke128",
        model_id="base",
        batch_size=2,
        count=128,
        speed=1.5,
        wall=90.0,
        resume=True,
        attempt_count=2,
    )
    for model_id in MODEL_IDS:
        for batch_size, speed, wall in ((1, 1.0, 100.0), (4, 1.6, 70.0)):
            contaminated = model_id == "base" and batch_size == 4
            _completed(
                tmp_path,
                config,
                stage="confirm512",
                model_id=model_id,
                batch_size=batch_size,
                count=512,
                speed=speed,
                wall=wall,
                resume=contaminated,
                attempt_count=2 if contaminated else 1,
            )
    shadow = MODEL_IDS[1]
    _completed(
        tmp_path,
        config,
        stage="shadow3841",
        model_id=shadow,
        batch_size=1,
        count=3841,
        speed=1.0,
        wall=100.0,
    )
    _completed(
        tmp_path,
        config,
        stage="shadow3841",
        model_id=shadow,
        batch_size=4,
        count=3841,
        speed=1.6,
        wall=70.0,
        token_offset=1,
    )
    report = audit_final(root=tmp_path, config=config, shadow_adapter_id=shadow)
    assert report["status"] == "FAIL"
    assert (
        report["detailed_gates"][
            "performance_runs_single_attempt_without_resume"
        ]
        is False
    )
    assert report["gates"]["resume_without_gap_or_duplicate"] is False


def test_contract_only_reports_no_gpu_access(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_identifiable_backend_qualification_v2.py"),
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(tmp_path),
            "--contract-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload["status"] == "READY"
    assert payload["gpu_accessed"] is False


def test_cli_rejects_stop_after_records_outside_fixed_smoke_probe(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_identifiable_backend_qualification_v2.py"),
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(tmp_path),
            "--stage",
            "confirm512",
            "--model-id",
            "base",
            "--batch-size",
            "2",
            "--stop-after-records",
            "64",
            "--contract-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "reserved for smoke128/base/batch2" in completed.stderr
