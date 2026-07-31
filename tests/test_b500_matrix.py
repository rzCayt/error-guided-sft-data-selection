import copy
import json
from pathlib import Path

import pytest

from eg_sft.experiment.b500_matrix import preflight_b500_matrix
from eg_sft.training.b500 import file_sha256, selected_id_sha256


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, missing_rds: bool = False) -> dict:
    protocol_path = tmp_path / "configs" / "protocol.json"
    recipe_path = tmp_path / "configs" / "recipe.json"
    execution_path = tmp_path / "configs" / "execution.json"
    runner_path = tmp_path / "scripts" / "runner.py"
    gate_path = tmp_path / "results" / "h1a" / "metrics.json"
    data_dir = tmp_path / "results" / "data"
    _write_json(protocol_path, {"version": "test"})
    _write_json(
        recipe_path,
        {
            "selection": {
                "allowed_strategies": ["random", "rds_all", "rds_error"],
                "budget": 500,
                "selection_seed": 20260722,
            },
            "formal_training_seeds": [17, 29, 41],
        },
    )
    _write_json(execution_path, {"version": "test"})
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("print('not executed')\n", encoding="utf-8")
    _write_json(gate_path, {"h1a_gate_passed": True})

    candidates = [
        {
            "candidate_id": f"candidate-{index:03d}",
            "source_index": index,
            "prompt_sha256": f"prompt-{index:03d}",
            "response_sha256": f"response-{index:03d}",
        }
        for index in range(500)
    ]
    _write_jsonl(data_dir / "tulu_candidate_pool.jsonl", candidates)
    _write_jsonl(data_dir / "gsm8k_records.jsonl", [{"record_id": "test"}])

    selections: dict[str, dict] = {}
    for strategy in ("random", "rds_all", "rds_error"):
        path = tmp_path / "results" / f"{strategy}_selection" / "manifest.json"
        selected = [
            {
                **candidate,
                "total_tokens": 10,
                "supervised_tokens": 4,
            }
            for candidate in candidates
        ]
        manifest = {
            "strategy": strategy,
            "budget": 500,
            "selection_seed": 20260722,
            "selected_id_sha256": selected_id_sha256(selected),
            "selected_candidates": selected,
        }
        if not (missing_rds and strategy != "random"):
            _write_json(path, manifest)
            sha256 = file_sha256(path)
        else:
            sha256 = None
        selections[strategy] = {
            "path": str(path.relative_to(tmp_path)).replace("\\", "/"),
            "sha256": sha256,
        }

    schedule = [
        {"strategy": strategy, "seed": seed}
        for seed in (17, 29, 41)
        for strategy in ("random", "rds_all", "rds_error")
    ]
    return {
        "matrix_version": "b500-formal-comparison-v1",
        "protocol_config": {
            "path": "configs/protocol.json",
            "sha256": file_sha256(protocol_path),
        },
        "recipe_config": {
            "path": "configs/recipe.json",
            "sha256": file_sha256(recipe_path),
        },
        "runner": {
            "path": "scripts/runner.py",
            "sha256": file_sha256(runner_path),
        },
        "execution_config": {
            "path": "configs/execution.json",
            "sha256": file_sha256(execution_path),
        },
        "data_manifest": {
            "directory": "results/data",
            "required_files": {
                "gsm8k_records.jsonl": file_sha256(data_dir / "gsm8k_records.jsonl"),
                "tulu_candidate_pool.jsonl": file_sha256(data_dir / "tulu_candidate_pool.jsonl"),
            },
        },
        "h1a_gate": {
            "path": "results/h1a/metrics.json",
            "sha256": file_sha256(gate_path),
            "required_field": "h1a_gate_passed",
            "required_value": True,
        },
        "output_root": "results/formal_runs",
        "formal_training_seeds": [17, 29, 41],
        "selections": selections,
        "job_order": schedule,
        "execution_policy": {
            "automatic_execution": False,
            "one_job_per_manual_invocation": True,
        },
    }


def test_ready_matrix_has_exactly_nine_jobs_and_one_common_contract(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    report = preflight_b500_matrix(
        spec=spec,
        repo_root=tmp_path,
        python_executable="PYTHON",
        matrix_config_path="configs/matrix.json",
    )
    assert report["status"] == "READY_FOR_MANUAL_ONE_JOB_AT_A_TIME"
    assert report["job_count"] == 9
    assert report["ready_selection_count"] == 3
    assert {(job["strategy"], job["seed"]) for job in report["jobs"]} == {
        (strategy, seed) for strategy in ("random", "rds_all", "rds_error") for seed in (17, 29, 41)
    }
    assert {job["common_contract_sha256"] for job in report["jobs"]} == {
        report["common_contract_sha256"]
    }
    assert all(job["status"] == "READY_FOR_MANUAL_INVOCATION" for job in report["jobs"])
    assert all(job["command"][0] == "PYTHON" for job in report["jobs"])
    assert all(
        job["command"][2:4]
        == [
            "--matrix-config",
            "configs/matrix.json",
        ]
        for job in report["jobs"]
    )


def test_missing_rds_manifests_block_the_whole_matrix(tmp_path: Path) -> None:
    spec = _fixture(tmp_path, missing_rds=True)
    report = preflight_b500_matrix(spec=spec, repo_root=tmp_path)
    assert report["status"] == "BLOCKED_INCOMPLETE_SELECTION_FREEZE"
    assert report["ready_selection_count"] == 1
    assert {(item["strategy"], item["status"]) for item in report["next_blockers"]} == {
        ("rds_all", "BLOCKED_MISSING_SELECTION_MANIFEST"),
        ("rds_error", "BLOCKED_MISSING_SELECTION_MANIFEST"),
    }
    random_jobs = [job for job in report["jobs"] if job["strategy"] == "random"]
    assert all(job["status"] == "BLOCKED_UNTIL_ALL_SELECTIONS_ARE_FROZEN" for job in random_jobs)


def test_existing_manifest_without_frozen_hash_is_blocked(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    spec["selections"]["rds_all"]["sha256"] = None
    report = preflight_b500_matrix(spec=spec, repo_root=tmp_path)
    assert report["selections"]["rds_all"]["status"] == "BLOCKED_UNFROZEN_SELECTION_SHA256"


def test_schedule_must_contain_every_pair_once(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    spec["job_order"][-1] = copy.deepcopy(spec["job_order"][0])
    with pytest.raises(ValueError, match="each strategy-seed pair"):
        preflight_b500_matrix(spec=spec, repo_root=tmp_path)


def test_h1a_gate_must_remain_passed(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    gate_path = tmp_path / spec["h1a_gate"]["path"]
    _write_json(gate_path, {"h1a_gate_passed": False})
    spec["h1a_gate"]["sha256"] = file_sha256(gate_path)
    with pytest.raises(ValueError, match="H1a gate"):
        preflight_b500_matrix(spec=spec, repo_root=tmp_path)


def test_repository_relative_paths_cannot_escape(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    spec["output_root"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        preflight_b500_matrix(spec=spec, repo_root=tmp_path)
