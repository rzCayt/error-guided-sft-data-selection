import hashlib
import json
from pathlib import Path

import pytest

from eg_sft.experiment.budget_equivalent_matrix import (
    phase1_registry,
    resolve_phase1_contract,
    validate_matrix_config,
)
from eg_sft.selection.budget_equivalent import build_selection_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    protocol = tmp_path / "protocol.json"
    recipe = tmp_path / "recipe.json"
    gates = tmp_path / "gates.json"
    protocol.write_text("{}", encoding="utf-8")
    recipe.write_text("{}", encoding="utf-8")
    gates.write_text(
        json.dumps(
            {
                "targeted_policy_gate_passed": True,
                "formal_near_duplicate_control": True,
            }
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "gsm8k_records.jsonl").write_text("x", encoding="utf-8")
    (data_dir / "tulu_candidate_pool.jsonl").write_text("y", encoding="utf-8")
    methods = [
        "random_free_mix",
        "rds_error_free_mix",
        "random_common_mix",
        "rds_error_common_mix",
    ]
    jobs = []
    selected = [
        {
            "candidate_id": f"c{i}",
            "supervised_tokens": 64,
            "total_tokens": 160,
        }
        for i in range(500)
    ]
    selected[0]["supervised_tokens"] = 32_000 - 64 * 499
    for replicate in range(1, 5):
        for method in methods:
            selection_seed = 1000 + replicate if method.startswith("random") else replicate * 101
            audit = {
                "selected_count": 500,
                "response_relative_error": 0.0,
                "duplicate_cluster_mode": "near_duplicate_cluster_manifest",
            }
            if method.endswith("common_mix"):
                audit.update(
                    {
                        "common_mix_quota_matches": True,
                        "prompt_relative_error": 0.0,
                        "total_relative_error": 0.0,
                    }
                )
            manifest = build_selection_manifest(
                method=method,
                selection_seed=selection_seed,
                train_seed=17,
                selected=selected,
                audit=audit,
                provenance={},
            )
            path = tmp_path / f"selection_{replicate}_{method}.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            jobs.append(
                {
                    "cell_id": f"rep{replicate}_{method}_train17",
                    "replicate_index": replicate,
                    "method": method,
                    "selection_seed": selection_seed,
                    "train_seed": 17,
                    "selection_manifest": {"path": path.name, "sha256": _sha(path)},
                }
            )
    config = {
        "phase1_protocol_version": "budget-equivalent-phase1-matrix-v3",
        "protocol_config": {"path": protocol.name, "sha256": _sha(protocol)},
        "base_recipe_config": {"path": recipe.name, "sha256": _sha(recipe)},
        "information_gates": {"path": gates.name, "sha256": _sha(gates)},
        "data_manifest": {
            "directory": "data",
            "required_files": {
                "gsm8k_records.jsonl": _sha(data_dir / "gsm8k_records.jsonl"),
                "tulu_candidate_pool.jsonl": _sha(data_dir / "tulu_candidate_pool.jsonl"),
            },
        },
        "methods": methods,
        "job_order": jobs,
        "output_root": ".aris/runs",
        "training": {
            "selection_budget": 500,
            "epochs": 2,
            "optimizer_steps": 64,
            "max_length": 512,
            "loss_normalization": "optimizer_step_response_token_sum_over_count",
            "single_training_process": True,
        },
        "evaluation": {"expected_record_count": 1319},
        "execution_policy": {
            "one_cell_per_invocation": True,
            "automatic_next_cell": False,
            "accuracy_blind_until_all_audits": True,
        },
    }
    config_path = tmp_path / "matrix.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, jobs[0]["cell_id"]


def test_matrix_has_exactly_sixteen_cells_and_resolves_one(tmp_path: Path) -> None:
    config_path, cell_id = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    validate_matrix_config(config)
    contract = resolve_phase1_contract(
        repo_root=tmp_path,
        config_path=config_path,
        cell_id=cell_id,
    )
    assert contract["cell_id"] == cell_id
    assert len(contract["selection"]["selected"]) == 500
    registry = phase1_registry(repo_root=tmp_path, config_path=config_path)
    assert registry["job_count"] == 16
    assert registry["audited_pass_count"] == 0


def test_matrix_rejects_exact_prompt_duplicate_fallback(tmp_path: Path) -> None:
    config_path, cell_id = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    first_path = tmp_path / config["job_order"][0]["selection_manifest"]["path"]
    manifest = json.loads(first_path.read_text())
    manifest["budget_audit"]["duplicate_cluster_mode"] = "exact_prompt_fallback"
    content = dict(manifest)
    content.pop("manifest_content_sha256")
    from eg_sft.selection.budget_equivalent import canonical_json_sha256

    manifest["manifest_content_sha256"] = canonical_json_sha256(content)
    first_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["job_order"][0]["selection_manifest"]["sha256"] = _sha(first_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbids exact-prompt"):
        resolve_phase1_contract(repo_root=tmp_path, config_path=config_path, cell_id=cell_id)
