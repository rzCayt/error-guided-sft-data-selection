from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from eg_sft.experiment.identifiable_budget_v4 import validate_identifiable_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_v4_schedule_has_four_dose_and_eight_common_jobs() -> None:
    payload = json.loads((ROOT / "configs" / "identifiable_budget_v4_matrix.json").read_text(encoding="utf-8"))
    validate_identifiable_matrix(payload)
    assert len(payload["job_order"]) == 12
    assert sum(row["study"] == "dose_only" for row in payload["job_order"]) == 4
    assert sum(row["study"] == "common_seed29" for row in payload["job_order"]) == 8
    gpu0 = {wave["gpu0"] for wave in payload["dual_gpu_schedule"]}
    gpu1 = {wave["gpu1"] for wave in payload["dual_gpu_schedule"]}
    assert len(gpu0) == len(gpu1) == 6
    assert not gpu0 & gpu1


def test_v4_rejects_changed_dose() -> None:
    payload = json.loads((ROOT / "configs" / "identifiable_budget_v4_matrix.json").read_text(encoding="utf-8"))
    payload["job_order"][0]["supervision_token_cap"] = 63681
    with pytest.raises(ValueError, match="dose-only jobs changed"):
        validate_identifiable_matrix(payload)


def test_v4_worker_id_defaults_only_when_environment_is_absent() -> None:
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        from run_identifiable_budget_v4_cell import resolve_worker_id
    finally:
        sys.path.remove(scripts)

    assert resolve_worker_id(None) == "manual"
    assert resolve_worker_id("gpu0") == "gpu0"
    assert resolve_worker_id(" gpu1 ") == "gpu1"
    with pytest.raises(ValueError, match="simple non-empty"):
        resolve_worker_id("")
    with pytest.raises(ValueError, match="simple non-empty"):
        resolve_worker_id("gpu/0")
