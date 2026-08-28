from __future__ import annotations

import json
from pathlib import Path

import pytest

from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract
from eg_sft.experiment.budget_equivalent_ood_runtime import resolve_ood_contract
from eg_sft.experiment.phase2_clean_common_v8 import validate_clean_common_matrix
from eg_sft.experiment.phase2_v7_control import Phase2StateStore, worker_schedule


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase2_clean_common24_v8_canonical.json"


def _payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v8_is_clean_balanced_24_cell_common_block() -> None:
    payload = _payload()
    validate_clean_common_matrix(payload)
    assert len(payload["job_order"]) == 24
    assert {row["train_seed"] for row in payload["job_order"]} == {17, 29, 41}
    assert {row["mix"] for row in payload["job_order"]} == {"common_mix"}
    assert payload["historical_cells_in_primary_analysis"] is False


def test_v8_workers_are_disjoint_and_balanced(tmp_path: Path) -> None:
    payload = _payload()
    gpu0 = worker_schedule(payload, "gpu0")
    gpu1 = worker_schedule(payload, "gpu1")
    assert len(gpu0) == len(gpu1) == 12
    assert not set(gpu0) & set(gpu1)
    store = Phase2StateStore(root=tmp_path / "control", matrix_path=CONFIG)
    report = store.initialize()
    assert report["job_count"] == report["created_count"] == 24


def test_v8_resolves_same_parent_selection_for_all_three_seeds() -> None:
    hashes = set()
    for seed in (17, 29, 41):
        contract = resolve_phase1_contract(
            repo_root=ROOT,
            config_path=CONFIG,
            cell_id=f"v8_rep1_random_common_mix_train{seed}",
        )
        hashes.add(contract["selection"]["file_sha256"])
        assert contract["seed"] == seed
        assert contract["study"] == "clean_new_environment_common_block"
    assert hashes == {
        "a2d0c1085ca0db7878f85d421c6f696ba19b3158555e11e8b9837f8c9d01fa42"
    }


def test_v8_rejects_free_mix_or_historical_primary() -> None:
    payload = _payload()
    payload["historical_cells_in_primary_analysis"] = True
    with pytest.raises(ValueError, match="historical"):
        validate_clean_common_matrix(payload)


def test_v8_rejects_wrong_seed_or_selection_hash_shape() -> None:
    payload = _payload()
    payload["job_order"][0]["train_seed"] = 999
    with pytest.raises(ValueError, match="crossing"):
        validate_clean_common_matrix(payload)
    payload = _payload()
    payload["job_order"][0]["parent_selection_manifest_sha256"] = "short"
    with pytest.raises(ValueError, match="binding"):
        validate_clean_common_matrix(payload)


def test_v8_ood_contract_reuses_frozen_data() -> None:
    svamp = resolve_ood_contract(
        repo_root=ROOT, matrix_config_path=CONFIG, dataset="svamp"
    )
    assert len(svamp["records"]) == 300
    assert svamp["matrix"]["phase2_extension"]["historical_seed17_external_only"] is True
