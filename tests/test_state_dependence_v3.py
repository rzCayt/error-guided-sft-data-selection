from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eg_sft.analysis.state_dependence import (
    bootstrap_interval,
    percentile,
    top_k_jaccard,
    u0_point_metrics,
)


ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = _load_script("freeze_state_dependence_panel_v3")
U0 = _load_script("analyze_state_dependence_u0_v3")
U1 = _load_script("analyze_state_dependence_u1_v3")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _manifest(candidate_ids: list[str]) -> dict:
    return {"selected_candidates": [{"candidate_id": value} for value in candidate_ids]}


def test_statistics_helpers() -> None:
    assert percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    interval = bootstrap_interval(
        sample_size=8,
        statistic=lambda indices: sum(indices) / len(indices),
        repetitions=100,
        seed=7,
    )
    assert interval["lower_95"] <= interval["median"] <= interval["upper_95"]
    ids = ["a", "b", "c", "d"]
    assert top_k_jaccard(ids, [4, 3, 2, 1], [4, 3, 1, 2], k=2) == 1.0
    metrics = u0_point_metrics(
        [[index, index + 0.01, index - 0.01] for index in range(1, 9)],
        [17, 29, 41],
    )
    assert metrics["icc_absolute_agreement_a1"] > 0.99
    assert metrics["minimum_pairwise_spearman"] == pytest.approx(1.0)


def test_freeze_panel_excludes_every_target_training_id(tmp_path: Path) -> None:
    states = [
        "v8_rep1_random_common_mix_train17",
        "v8_rep1_rds_error_common_mix_train17",
        "v8_rep4_random_common_mix_train17",
        "v8_rep4_rds_error_common_mix_train17",
    ]
    protocol = tmp_path / "protocol.json"
    _write_json(
        protocol,
        {"stage_u1_cross_state_transfer": {"initial_adapter_states": states}},
    )
    rows = []
    for index in range(96):
        rows.append(
            {
                "candidate_id": f"c{index:03d}",
                "source_dataset": f"source{index % 5}",
                "source_index": index,
                "prompt_sha256": f"{index:064x}",
                "error_query_rank": index,
                "error_query_score": 1.0 - index / 96,
                "all_query_rank": index,
                "all_query_score": 1.0 - index / 96,
                "training_supervised_tokens": 8 + index,
                "training_total_tokens": 16 + index,
                "response_only_trainable": True,
            }
        )
    scores = tmp_path / "scores.jsonl"
    _write_jsonl(scores, rows)
    manifest_root = tmp_path / "manifests"
    adapters = []
    seen = [f"c{index:03d}" for index in (0, 1, 24, 25, 48, 49, 72, 73)]
    for state_id in states:
        method = "random_common_mix" if "random" in state_id else "rds_error_common_mix"
        replicate = 1 if "rep1" in state_id else 4
        path = manifest_root / f"replicate_{replicate:02d}" / method / "selection_manifest.json"
        selected = seen + [f"outside{index:03d}" for index in range(492)]
        _write_json(path, _manifest(selected))
        adapters.append(
            {
                "cell_id": state_id,
                "method": method,
                "replicate_index": replicate,
                "selection_manifest_sha256": FREEZE.file_sha256(path),
            }
        )
    adapter_index = tmp_path / "adapter_index.json"
    _write_json(adapter_index, {"adapters": adapters})
    panel_path = tmp_path / "panel.json"
    overlap_path = tmp_path / "overlap.json"
    panel, overlap = FREEZE.freeze_panel(
        protocol_path=protocol,
        candidate_scores_path=scores,
        adapter_index_path=adapter_index,
        manifest_root=manifest_root,
        output_panel_path=panel_path,
        output_overlap_path=overlap_path,
    )
    panel_ids = {row["candidate_id"] for row in panel["candidates"]}
    assert panel_ids.isdisjoint(seen)
    assert overlap["score_panel_seen_count"] == 8
    assert overlap["frozen_panel_overlap_count"] == 0
    assert panel["quartile_counts"] == {"q0": 12, "q1": 12, "q2": 12, "q3": 12}


def _analysis_protocol(states: list[str]) -> dict:
    return {
        "stage_u0a_fixed_state_reliability": {
            "probe_seeds": [17, 29, 41],
            "new_measurements": 144,
            "bootstrap_repetitions": 100,
            "go_gate": {
                "minimum_icc_a1_point": 0.75,
                "minimum_icc_a1_ci_lower": 0.60,
                "minimum_median_pairwise_spearman": 0.75,
                "minimum_pairwise_spearman": 0.65,
            },
            "stop_gate": {
                "icc_a1_below": 0.60,
                "any_pairwise_spearman_below": 0.50,
            },
        },
        "stage_u1_cross_state_transfer": {
            "probe_seeds": [17, 41],
            "initial_adapter_states": states,
            "bootstrap_repetitions": 100,
            "state_dependence_gate": {
                "minimum_states_with_spearman_below_0_75": 3,
                "maximum_state_ci_upper": 0.90,
                "maximum_median_top12_jaccard": 0.50,
                "requires_two_probe_seed_direction_agreement": True,
            },
            "state_stability_gate": {
                "minimum_all_state_spearman": 0.85,
                "minimum_all_state_ci_lower": 0.70,
                "minimum_all_state_top12_jaccard": 0.67,
                "maximum_all_state_practical_sign_flip": 0.10,
            },
            "ambiguous_action": "expand",
        },
    }


def test_u0_and_u1_v3_use_only_new_two_seed_audited_measurements(tmp_path: Path) -> None:
    ids = [f"c{index:02d}" for index in range(48)]
    states = [f"state{index}" for index in range(4)]
    protocol = tmp_path / "protocol.json"
    _write_json(protocol, _analysis_protocol(states))
    panel = tmp_path / "panel.json"
    _write_json(panel, {"candidates": [{"candidate_id": value} for value in ids]})
    u0_run = tmp_path / "u0_run"
    _write_json(
        u0_run / "INDEPENDENT_AUDIT.json",
        {
            "schema_version": "candidate-utility-state-probe-audit-v3",
            "status": "PASS",
            "state_id": "zero_initialized_lora",
            "historical_measurements_reused": False,
        },
    )
    u0_rows = []
    for index, candidate_id in enumerate(ids):
        for seed, shift in ((17, 0.0), (29, 0.00001), (41, -0.00001)):
            u0_rows.append(
                {"candidate_id": candidate_id, "probe_seed": seed, "utility": index / 1000 + shift}
            )
    _write_jsonl(u0_run / "utility_measurements.jsonl", u0_rows)
    u0_root = tmp_path / "u0_result"
    u0 = U0.analyze(
        protocol_path=protocol,
        panel_path=panel,
        run_dir=u0_run,
        output_root=u0_root,
    )
    assert u0["status"] == "GO"
    assert u0["historical_measurements_reused"] is False

    state_dirs = []
    for state_id in states:
        run = tmp_path / state_id
        _write_json(
            run / "INDEPENDENT_AUDIT.json",
            {
                "schema_version": "candidate-utility-state-probe-audit-v3",
                "status": "PASS",
                "state_id": state_id,
            },
        )
        rows = []
        for index, candidate_id in enumerate(ids):
            for seed, shift in ((17, 0.0), (41, 0.00001)):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "probe_seed": seed,
                        "utility": index / 1000 + shift,
                    }
                )
        _write_jsonl(run / "utility_measurements.jsonl", rows)
        state_dirs.append(run)
    u1 = U1.analyze(
        protocol_path=protocol,
        u0_root=u0_root,
        state_run_dirs=state_dirs,
        output_root=tmp_path / "u1_result",
    )
    assert u1["status"] == "STATE_STABILITY"
    assert u1["probe_seeds"] == [17, 41]
