import json
from pathlib import Path

from eg_sft.experiment.budget_equivalent_protocol import (
    phase1_jobs,
    preflight_protocol,
    validate_protocol_config,
)


def _payload() -> dict:
    return {
        "protocol_version": "budget-equivalent-selection-v3",
        "base_commit": "6d5bcea",
        "protocol_config": {"path": "protocol.json", "sha256": "a" * 64},
        "candidate_inventory": {"path": "candidates.jsonl", "sha256": "b" * 64},
        "query_inventory": {"path": "queries.jsonl", "sha256": "c" * 64},
        "similarity_artifact": {"path": ".aris/similarity.pt", "sha256": None},
        "near_duplicate_clusters": {"path": ".aris/clusters.jsonl", "sha256": None},
        "selection": {
            "selected_example_count": 500,
            "target_response_supervision_tokens": 32000,
            "response_tolerance_fraction": 0.005,
            "common_prompt_tolerance_fraction": 0.01,
            "common_total_tolerance_fraction": 0.01,
            "requested_response_length_bins": 5,
            "minimum_source_quota": 4,
            "minimum_freedom_ratio": 4.0,
            "maximum_forced_selected_fraction": 0.10,
            "selection_replicate_seeds": [101, 202, 303, 404],
            "random_priority_seeds": [1101, 1202, 1303, 1404],
            "phase1_train_seed": 17,
        },
        "information_gates": {},
        "methods": [
            "random_free_mix",
            "rds_error_free_mix",
            "random_common_mix",
            "rds_error_common_mix",
        ],
        "output_root": ".aris/output",
    }


def test_phase1_matrix_contains_exactly_sixteen_unique_cells() -> None:
    payload = _payload()
    validate_protocol_config(payload)
    jobs = phase1_jobs(payload)
    assert len(jobs) == 16
    assert len({job["cell_id"] for job in jobs}) == 16
    assert {job["train_seed"] for job in jobs} == {17}


def test_preflight_reports_missing_or_unfrozen_inputs_without_claiming_ready(
    tmp_path: Path,
) -> None:
    payload = _payload()
    for name in ("protocol_config", "candidate_inventory", "query_inventory"):
        path = tmp_path / payload[name]["path"]
        path.write_text(name, encoding="utf-8")
        import hashlib

        payload[name]["sha256"] = hashlib.sha256(name.encode()).hexdigest()
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    report = preflight_protocol(repo_root=tmp_path, config_path=config)
    assert report["status"] == "BLOCKED"
    assert report["phase1_job_count"] == 16
    assert report["bindings"]["similarity_artifact"]["status"] == "BLOCKED_MISSING"
    assert report["formal_selection_permitted"] is False
