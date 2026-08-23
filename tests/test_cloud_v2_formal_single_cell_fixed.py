import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eg_sft.evaluation.formal_two_worker import (
    formal_shards,
    merge_formal_worker_outputs,
    records_for_formal_shard,
    validate_formal_worker_prefix,
)
from eg_sft.experiment.cloud_v2_formal import (
    build_formal_registry,
    engineering_stdout_payload,
    resolve_formal_contract,
    validate_formal_config,
)
from eg_sft.training.effective_batch import should_write_checkpoint


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (
            _root() / "configs" / "cloud_v2_formal_b500_single_cell_fixed_v1.json"
        ).read_text(encoding="utf-8")
    )


def _records() -> list[dict]:
    return [{"record_id": f"test-{index:04d}"} for index in range(1319)]


def _worker_payloads() -> tuple[tuple, dict[str, dict]]:
    records = _records()
    shards = formal_shards(_config()["evaluation"])
    payloads = {}
    for shard in shards:
        rows = [
            {"record_id": row["record_id"], "shard_id": shard.shard_id}
            for row in records_for_formal_shard(records, shard)
        ]
        payloads[shard.shard_id] = {
            "manifest": {
                "worker": {
                    "shard_id": shard.shard_id,
                    "start_index": shard.start_index,
                    "end_index": shard.end_index,
                    "physical_batch_size": 1,
                }
            },
            "metrics": {
                "status": "PASS",
                "adapter_model_sha256": "a" * 64,
                "gpu_uuid": "GPU-formal",
                "raw_outputs_sha256": "b" * 64,
                "model_load_seconds": 1.0,
                "generation_seconds": 2.0,
                "peak_allocated_memory_gib": 8.0,
                "peak_reserved_memory_gib": 9.0,
            },
            "rows": rows,
        }
    return shards, payloads


def test_fixed_config_enforces_single_training_batch_and_batch_one_eval() -> None:
    config = _config()
    validate_formal_config(config)
    assert config["training"]["micro_batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["training"]["checkpoint_every_optimizer_steps"] == 10
    assert config["evaluation"]["physical_batch_size_per_worker"] == 1
    changed = copy.deepcopy(config)
    changed["evaluation"]["physical_batch_size_per_worker"] = 2
    with pytest.raises(ValueError, match="batch size one"):
        validate_formal_config(changed)


def test_fixed_registry_has_required_order_and_never_auto_runs() -> None:
    config_path = _root() / "configs" / "cloud_v2_formal_b500_single_cell_fixed_v1.json"
    registry = build_formal_registry(repo_root=_root(), config_path=config_path)
    assert registry["automatic_execution"] is False
    assert registry["job_count"] == 9
    assert [(row["method"], row["seed"]) for row in registry["jobs"]] == [
        ("rds_all", 17),
        ("rds_error", 17),
        ("rds_all", 29),
        ("rds_error", 29),
        ("rds_all", 41),
        ("rds_error", 41),
        ("random", 17),
        ("random", 29),
        ("random", 41),
    ]
    assert all("--method" in row["command"] and "--seed" in row["command"] for row in registry["jobs"])


def test_real_frozen_selection_hashes_resolve_for_each_method() -> None:
    config_path = _root() / "configs" / "cloud_v2_formal_b500_single_cell_fixed_v1.json"
    for method in ("random", "rds_all", "rds_error"):
        contract = resolve_formal_contract(
            repo_root=_root(),
            config_path=config_path,
            method=method,
            seed=17,
        )
        assert len(contract["selection"]["selected"]) == 500
        assert len(contract["selection"]["selected_id_sha256"]) == 64


def test_checkpoint_schedule_is_every_ten_and_final_sixty_three() -> None:
    saved = [
        step
        for step in range(1, 64)
        if should_write_checkpoint(
            optimizer_step=step,
            optimizer_steps_planned=63,
            checkpoint_every_optimizer_steps=10,
        )
    ]
    assert saved == [10, 20, 30, 40, 50, 60, 63]


def test_formal_workers_merge_exactly_1319_in_original_order() -> None:
    shards, payloads = _worker_payloads()
    merged, report = merge_formal_worker_outputs(
        frozen_records=_records(), shards=shards, worker_payloads=payloads
    )
    assert report["status"] == "PASS"
    assert report["record_count"] == 1319
    assert [row["record_id"] for row in merged] == [row["record_id"] for row in _records()]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "worker_fail", "gpu_mismatch"])
def test_formal_merge_rejects_any_incomplete_or_inconsistent_worker(fault: str) -> None:
    shards, payloads = _worker_payloads()
    if fault == "missing":
        payloads["test_shard1"]["rows"].pop()
    elif fault == "duplicate":
        payloads["test_shard0"]["rows"][1]["record_id"] = payloads["test_shard0"][
            "rows"
        ][0]["record_id"]
    elif fault == "worker_fail":
        payloads["test_shard0"]["metrics"]["status"] = "FAIL"
    else:
        payloads["test_shard1"]["metrics"]["gpu_uuid"] = "GPU-other"
    with pytest.raises(ValueError):
        merge_formal_worker_outputs(
            frozen_records=_records(), shards=shards, worker_payloads=payloads
        )


def test_formal_worker_prefix_supports_resume_without_gap() -> None:
    shard = formal_shards(_config()["evaluation"])[0]
    frozen = records_for_formal_shard(_records(), shard)
    prefix = [
        {"record_id": row["record_id"], "shard_id": shard.shard_id}
        for row in frozen[:29]
    ]
    assert validate_formal_worker_prefix(
        rows=prefix, frozen_shard_records=frozen, shard_id=shard.shard_id
    ) == 29
    prefix[-1]["record_id"] = frozen[30]["record_id"]
    with pytest.raises(ValueError, match="not a frozen prefix"):
        validate_formal_worker_prefix(
            rows=prefix, frozen_shard_records=frozen, shard_id=shard.shard_id
        )


def test_engineering_stdout_contains_no_result_metric() -> None:
    payload = engineering_stdout_payload(
        status="COMPLETE",
        run_id="run",
        hashes={"raw_outputs_sha256": "a" * 64},
        stage="formal_cell",
    )
    serialized = json.dumps(payload, sort_keys=True)
    assert "numeric_accuracy" not in serialized
    assert "method_delta" not in serialized
    assert payload["next_cell_started"] is False


def test_fixed_formal_clis_are_cpu_safe_for_help() -> None:
    scripts = (
        "run_cloud_v2_formal_cell_fixed.py",
        "audit_cloud_v2_formal_cell_fixed.py",
        "preflight_cloud_v2_formal_matrix.py",
    )
    for script in scripts:
        process = subprocess.run(
            [sys.executable, str(_root() / "scripts" / script), "--help"],
            cwd=_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stderr
