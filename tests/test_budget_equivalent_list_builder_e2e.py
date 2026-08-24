import hashlib
import json
from pathlib import Path

import torch

import eg_sft.experiment.budget_equivalent_lists as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_engineering_builder_produces_four_real_replicates_and_sixteen_lists(
    tmp_path: Path, monkeypatch
) -> None:
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
    similarity_path = tmp_path / "similarity.pt"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(query_path, queries)
    protocol_path.write_text("{}", encoding="utf-8")
    torch.save(
        {
            "similarity": torch.randn(10, 80, generator=torch.Generator().manual_seed(7)),
            "query_ids": [row["record_id"] for row in queries],
            "candidate_ids": [row["candidate_id"] for row in candidates],
        },
        similarity_path,
    )
    config = {
        "protocol_version": "budget-equivalent-selection-v3",
        "protocol_config": {"path": "protocol.json", "sha256": _sha(protocol_path)},
        "candidate_inventory": {"path": "candidates.jsonl", "sha256": _sha(candidate_path)},
        "query_inventory": {"path": "queries.jsonl", "sha256": _sha(query_path)},
        "similarity_artifact": {"path": "similarity.pt", "sha256": _sha(similarity_path)},
        "near_duplicate_clusters": {"path": "missing.jsonl", "sha256": None},
        "selection": {
            "selected_example_count": 20,
            "target_response_supervision_tokens": 170,
            "response_tolerance_fraction": 0.05,
            "common_prompt_tolerance_fraction": 0.05,
            "common_total_tolerance_fraction": 0.05,
            "requested_response_length_bins": 4,
            "minimum_source_quota": 2,
            "minimum_freedom_ratio": 2.0,
            "maximum_forced_selected_fraction": 0.10,
            "selection_replicate_seeds": [101, 202, 303, 404],
            "random_priority_seeds": [1101, 1202, 1303, 1404],
            "phase1_train_seed": 17,
        },
        "information_gates": {
            "minimum_error_vs_all_top500_changed_fraction": 0.0,
            "maximum_error_vs_all_rank_spearman": 1.01,
            "minimum_error_selection_stability_jaccard": 0.0,
        },
        "methods": list(builder.CORE_METHODS),
        "output_root": ".aris/output",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(builder, "validate_protocol_config", lambda _: None)
    monkeypatch.setattr(
        builder,
        "preflight_protocol",
        lambda **_: {
            "bindings": {
                "protocol_config": {"status": "READY"},
                "candidate_inventory": {"status": "READY"},
                "query_inventory": {"status": "READY"},
                "similarity_artifact": {"status": "READY"},
                "near_duplicate_clusters": {"status": "BLOCKED_MISSING"},
            }
        },
    )
    output = tmp_path / "output"
    index = builder.build_phase1_lists(
        repo_root=tmp_path,
        config_path=config_path,
        output_root=output,
        engineering_allow_exact_prompt_fallback=True,
    )
    assert index["selection_count"] == 16
    assert len(list(output.glob("replicate_*/**/selection_manifest.json"))) == 16
    gates = json.loads((output / "information_gates.json").read_text())
    assert gates["replicate_count"] == 4
    assert gates["formal_near_duplicate_control"] is False
    assert gates["phase1_core_matrix_permitted"] is False
