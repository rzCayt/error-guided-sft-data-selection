import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eg_sft.evaluation.two_worker_calibration import (
    compare_to_single_worker_reference,
    merge_worker_outputs,
    records_for_shard,
    validate_two_worker_config,
    validate_worker_prefix,
)


def _config() -> dict:
    return {
        "calibration_version": "cloud-v2-two-worker-generation-v1",
        "worker_count": 2,
        "physical_batch_size_per_worker": 1,
        "cuda_device_index": 0,
        "expected_record_count": 128,
        "shards": [
            {"shard_id": "shard0", "start_index": 0, "end_index": 64},
            {"shard_id": "shard1", "start_index": 64, "end_index": 128},
        ],
    }


def _records() -> list[dict]:
    return [{"record_id": f"r{index:03d}"} for index in range(128)]


def _row(record: dict, shard_id: str) -> dict:
    value = record["record_id"][1:]
    return {
        "record_id": record["record_id"],
        "shard_id": shard_id,
        "raw_output": f"Final answer: {value}",
        "parse_status": "ok",
        "parsed_prediction": value,
        "numeric_correct": True,
    }


def _payloads() -> tuple[tuple, dict[str, dict]]:
    records = _records()
    shards = validate_two_worker_config(_config())
    payloads = {}
    for shard in shards:
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
                "gpu_uuid": "GPU-fixed",
                "model_load_seconds": 2.0,
                "generation_seconds": 10.0,
                "worker_wall_seconds": 12.0,
                "peak_allocated_memory_gib": 8.0,
                "peak_reserved_memory_gib": 9.0,
                "resume_invocation_count": 1,
            },
            "rows": [
                _row(record, shard.shard_id)
                for record in records_for_shard(records, shard)
            ],
        }
    return shards, payloads


def test_exact_workers_merge_and_match_reference() -> None:
    shards, payloads = _payloads()
    merged, report = merge_worker_outputs(
        frozen_records=_records(), shards=shards, worker_payloads=payloads
    )
    assert report["status"] == "PASS"
    assert [row["record_id"] for row in merged] == [row["record_id"] for row in _records()]
    comparison = compare_to_single_worker_reference(
        reference_rows=copy.deepcopy(merged), candidate_rows=merged
    )
    assert comparison["status"] == "PASS"


@pytest.mark.parametrize("failure_kind", ["missing", "duplicate", "shuffled", "worker_fail"])
def test_invalid_worker_artifact_cannot_produce_pass_merge(failure_kind: str) -> None:
    shards, payloads = _payloads()
    if failure_kind == "missing":
        payloads["shard1"]["rows"].pop()
    elif failure_kind == "duplicate":
        payloads["shard0"]["rows"][1]["record_id"] = payloads["shard0"]["rows"][0][
            "record_id"
        ]
    elif failure_kind == "shuffled":
        rows = payloads["shard1"]["rows"]
        rows[0], rows[1] = rows[1], rows[0]
    else:
        payloads["shard0"]["metrics"]["status"] = "FAIL"
    with pytest.raises(ValueError):
        merge_worker_outputs(
            frozen_records=_records(), shards=shards, worker_payloads=payloads
        )


def test_prefix_resume_accepts_only_deterministic_shard_prefix() -> None:
    shard = validate_two_worker_config(_config())[0]
    frozen = records_for_shard(_records(), shard)
    prefix = [_row(record, shard.shard_id) for record in frozen[:17]]
    assert validate_worker_prefix(
        rows=prefix, frozen_shard_records=frozen, shard_id=shard.shard_id
    ) == 17
    prefix[-1]["record_id"] = frozen[18]["record_id"]
    with pytest.raises(ValueError, match="not its frozen prefix"):
        validate_worker_prefix(
            rows=prefix,
            frozen_shard_records=frozen,
            shard_id=shard.shard_id,
        )


def test_prediction_difference_fails_exact_equivalence() -> None:
    shards, payloads = _payloads()
    merged, _ = merge_worker_outputs(
        frozen_records=_records(), shards=shards, worker_payloads=payloads
    )
    reference = copy.deepcopy(merged)
    merged[70].update(
        {
            "raw_output": "Final answer: 999",
            "parsed_prediction": "999",
            "numeric_correct": False,
        }
    )
    comparison = compare_to_single_worker_reference(
        reference_rows=reference, candidate_rows=merged
    )
    assert comparison["status"] == "FAIL"
    assert comparison["field_difference_counts"]["parsed_prediction"] == 1


def test_fixed_config_uses_git_ignored_runtime_root_and_fixed_clis_are_cpu_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "cloud_v2_two_worker_generation_fixed_v1.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    validate_two_worker_config(payload)
    assert payload["output_root"].startswith(".aris/")
    scripts = (
        "run_cloud_v2_generation_worker_fixed.py",
        "run_cloud_v2_two_worker_generation_fixed.py",
        "analyze_cloud_v2_two_worker_generation_fixed.py",
    )
    for script in scripts:
        process = subprocess.run(
            [sys.executable, str(root / "scripts" / script), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stderr
