from __future__ import annotations

from pathlib import Path

import pytest

from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract
from eg_sft.experiment.budget_equivalent_ood_runtime import resolve_ood_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "identifiable_budget_v4_matrix.json"


def _runtime_inputs_available() -> bool:
    return (
        ROOT
        / ".aris"
        / "compute"
        / "budget_equivalent_v3_selections"
        / "information_gates.json"
    ).is_file()


@pytest.mark.skipif(not _runtime_inputs_available(), reason="git-ignored frozen runtime inputs are absent")
def test_v4_contract_reuses_parent_selection_and_changes_only_declared_intervention() -> None:
    dose = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=CONFIG,
        cell_id="dose_rep1_random_free_mix_train17_cap63680",
    )
    common = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=CONFIG,
        cell_id="rep1_random_common_mix_train29",
    )
    assert dose["method"] == "random_free_mix"
    assert dose["seed"] == 17
    assert dose["supervision_token_cap"] == 63680
    assert dose["token_cap_policy"] == "hash_uniform_v1"
    assert dose["selection"]["file_sha256"] == "e4004d8721719c7a80e52de3e897a4638b53a782f15ccb0cf8220723ad3954f3"
    assert common["method"] == "random_common_mix"
    assert common["seed"] == 29
    assert common["supervision_token_cap"] is None
    assert common["selection"]["file_sha256"] == "a2d0c1085ca0db7878f85d421c6f696ba19b3158555e11e8b9837f8c9d01fa42"


@pytest.mark.skipif(not _runtime_inputs_available(), reason="git-ignored frozen runtime inputs are absent")
def test_v4_ood_contract_reuses_frozen_ood_membership() -> None:
    svamp = resolve_ood_contract(
        repo_root=ROOT,
        matrix_config_path=CONFIG,
        dataset="svamp",
    )
    assert len(svamp["records"]) == 300
    assert svamp["matrix"]["identifiable_budget_extension"]["matrix_version"] == "identifiable-budget-v4-extension-v1"
