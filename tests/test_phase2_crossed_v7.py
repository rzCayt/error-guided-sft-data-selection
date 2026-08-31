from __future__ import annotations

import json
from pathlib import Path

import pytest

from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract
from eg_sft.experiment.budget_equivalent_ood_runtime import resolve_ood_contract
from eg_sft.experiment.phase2_crossed_v7 import validate_phase2_matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase2_crossed_48cell_v7.json"


def _payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_phase2_has_balanced_32_cell_extension() -> None:
    payload = _payload()
    validate_phase2_matrix(payload)
    assert len(payload["job_order"]) == 32
    assert len(payload["dual_gpu_schedule"]) == 16
    assert set(row["train_seed"] for row in payload["job_order"]) == {29, 41}


def test_phase2_rejects_missing_or_unbalanced_cell() -> None:
    payload = _payload()
    payload["job_order"].pop()
    with pytest.raises(ValueError, match="32"):
        validate_phase2_matrix(payload)


def test_phase2_rejects_selection_hash_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["job_order"][0]["parent_selection_manifest_sha256"] = "0" * 64
    validate_phase2_matrix(payload)
    changed = tmp_path / "changed_phase2.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="parent selection"):
        resolve_phase1_contract(
            repo_root=ROOT,
            config_path=changed,
            cell_id=payload["job_order"][0]["cell_id"],
        )


def test_phase2_contract_changes_only_train_seed() -> None:
    contract = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=CONFIG,
        cell_id="rep1_random_common_mix_train29",
    )
    assert contract["method"] == "random_common_mix"
    assert contract["seed"] == 29
    assert contract["parent_cell_id"] == "rep1_random_common_mix_train17"
    assert contract["supervision_token_cap"] is None
    assert contract["selection"]["file_sha256"] == (
        "a2d0c1085ca0db7878f85d421c6f696ba19b3158555e11e8b9837f8c9d01fa42"
    )


def test_phase2_ood_contract_reuses_frozen_membership() -> None:
    svamp = resolve_ood_contract(
        repo_root=ROOT,
        matrix_config_path=CONFIG,
        dataset="svamp",
    )
    assert len(svamp["records"]) == 300
    assert svamp["matrix"]["phase2_extension"]["matrix_version"] == (
        "phase2-crossed-48cell-v7"
    )
