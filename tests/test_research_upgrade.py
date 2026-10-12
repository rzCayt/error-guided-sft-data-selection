from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from eg_sft.experiment.research_upgrade import (
    plan_batch, source_identity, validate_budget, validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def ledger() -> dict:
    return json.loads((ROOT / "configs/research_upgrade_budget.json").read_text(encoding="utf-8"))


def profile() -> dict:
    return {"schema_version": "compute-profile-v1", "status": "measured", "gpu_model": "fixture GPU",
            "work_unit": "one probe", "measurement_sha256": "a" * 64, "config_sha256": "b" * 64,
            "measured_at": "2026-10-13T10:00:00+08:00", "wall_seconds_per_unit": "60"}


def estimate(**kwargs) -> dict:
    inputs = {"ledger": ledger(), "profile": profile(), "phase": "weeks_1_3", "batch_id": "test",
              "units": 144, "hourly_cny": "2", "quote_at": "2026-10-13T11:00:00+08:00",
              "setup_hours": "0.2", "storage_cny": "1", "margin": "1.25"}
    inputs.update(kwargs)
    return plan_batch(**inputs)


def test_price_includes_setup_storage_and_margin() -> None:
    result = estimate()
    assert Decimal(result["reserve_cny"]) == Decimal("7.50")
    assert result["launch_authorized"] is False


def test_reservations_and_actual_charges_are_counted_once() -> None:
    value = ledger()
    value["entries"] = [
        {"batch_id": "a", "phase": "weeks_1_3", "status": "reserved",
         "reserved_cny": "100", "charged_cny": "0"},
        {"batch_id": "b", "phase": "weeks_1_3", "status": "settled",
         "reserved_cny": "0", "charged_cny": "50"},
    ]
    assert validate_budget(value)["weeks_1_3"] == Decimal(150)
    invalid = copy.deepcopy(value)
    invalid["entries"][1]["reserved_cny"] = "50"
    with pytest.raises(ValueError, match="Release reservation"):
        validate_budget(invalid)
    invalid = copy.deepcopy(value)
    invalid["entries"][1]["batch_id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        validate_budget(invalid)


@pytest.mark.parametrize("kwargs", [
    {"hourly_cny": "NaN"}, {"hourly_cny": "-1"}, {"units": 0}, {"margin": "0.9"},
    {"quote_at": "2026-10-13T11:00:00"}, {"phase": "unknown"},
    {"hourly_cny": "2000"}, {"profile": {**profile(), "status": "estimated"}},
])
def test_invalid_or_over_budget_batches_fail(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        estimate(**kwargs)


def test_pending_studies_cannot_pose_as_completed_gpu_evidence() -> None:
    value = json.loads((ROOT / "results/research_upgrade/registry.json").read_text(encoding="utf-8"))
    validate_registry(value, ROOT)
    value["studies"][0]["status"] = "completed"
    with pytest.raises(ValueError, match="actual result evidence"):
        validate_registry(value, ROOT)
    value["studies"][0]["status"] = "blocked_input"
    value["studies"][0]["gpu_result_available"] = True
    with pytest.raises(ValueError, match="GPU result flag"):
        validate_registry(value, ROOT)


def test_frozen_evidence_drift_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text("changed", encoding="utf-8")
    value = json.loads((ROOT / "results/research_upgrade/registry.json").read_text(encoding="utf-8"))
    for row in value["studies"]:
        row["frozen_inputs"] = []
    value["studies"][0]["frozen_inputs"] = [{"path": "evidence.json", "sha256": "a" * 64}]
    with pytest.raises(ValueError, match="Frozen input changed"):
        validate_registry(value, tmp_path)


def test_deployed_code_identity_detects_modified_source(tmp_path: Path) -> None:
    content = b"print('fixture')\n"
    source = tmp_path / "main.py"
    source.write_bytes(content)
    manifest = {"schema_version": "autodl-source-bundle-v1", "git_commit": "fixture",
                "files": [{"path": "main.py", "bytes": len(content),
                           "sha256": hashlib.sha256(content).hexdigest()}]}
    (tmp_path / "BUNDLE.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert source_identity(tmp_path)["kind"] == "verified_source_bundle"
    source.write_bytes(b"print('changed')\n")
    with pytest.raises(ValueError, match="Bundle source"):
        source_identity(tmp_path)
