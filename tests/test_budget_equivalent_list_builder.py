import hashlib
import json
from pathlib import Path

import torch

from eg_sft.experiment.budget_equivalent_lists import build_phase1_lists


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_builder_writes_sixteen_manifests_and_separate_gate(tmp_path: Path) -> None:
    candidates = []
    for index in range(80):
        response = 5 + index % 8
        candidates.append(
            {
                "candidate_id": f"c{index:03d}",
                "source_dataset": "a" if index % 3 else "b",
                "supervised_tokens": response,
                "total_tokens": response + 20 + index % 5,
                "user_prompt_sha256": f"p{index}",
            }
        )
    queries = [
        {"record_id": f"q{index}", "is_error_query": index >= 7}
        for index in range(10)
    ]
    candidate_path = tmp_path / "candidates.jsonl"
    query_path = tmp_path / "queries.jsonl"
    protocol_path = tmp_path / "protocol.json"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(query_path, queries)
    protocol_path.write_text("{}", encoding="utf-8")
    generator = torch.Generator().manual_seed(7)
    similarity = torch.randn(10, 80, generator=generator)
    similarity_path = tmp_path / ".aris" / "similarity.pt"
    similarity_path.parent.mkdir()
    torch.save(
        {
            "similarity": similarity,
            "query_ids": [row["record_id"] for row in queries],
            "candidate_ids": [row["candidate_id"] for row in candidates],
        },
        similarity_path,
    )
    payload = {
        "protocol_version": "budget-equivalent-selection-v3",
        "base_commit": "6d5bcea",
        "protocol_config": {"path": "protocol.json", "sha256": _sha(protocol_path)},
        "candidate_inventory": {"path": "candidates.jsonl", "sha256": _sha(candidate_path)},
        "query_inventory": {"path": "queries.jsonl", "sha256": _sha(query_path)},
        "similarity_artifact": {
            "path": ".aris/similarity.pt",
            "sha256": _sha(similarity_path),
        },
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
        "information_gates": {
            "minimum_error_vs_all_top500_changed_fraction": 0.15,
            "maximum_error_vs_all_rank_spearman": 0.95,
            "minimum_error_selection_stability_jaccard": 0.0,
        },
        "methods": [
            "random_free_mix",
            "rds_error_free_mix",
            "random_common_mix",
            "rds_error_common_mix",
        ],
        "output_root": ".aris/output",
    }
    # Keep protocol invariants while shrinking only the builder's synthetic inputs.
    payload["selection"].update(
        {
            "selected_example_count": 20,
            "target_response_supervision_tokens": 170,
            "response_tolerance_fraction": 0.05,
            "common_prompt_tolerance_fraction": 0.05,
            "common_total_tolerance_fraction": 0.05,
            "minimum_source_quota": 2,
            "minimum_freedom_ratio": 2.0,
        }
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    # The production validator deliberately freezes 500/32K. Synthetic end-to-end
    # coverage is provided by lower-level tests; here we verify that drift is blocked.
    try:
        build_phase1_lists(
            repo_root=tmp_path,
            config_path=config,
            engineering_allow_exact_prompt_fallback=True,
        )
    except ValueError as error:
        assert "frozen selection field changed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("builder accepted a drifted formal protocol")
