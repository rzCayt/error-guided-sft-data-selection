import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_selector_identifiability.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("audit_selector_identifiability", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _candidate(index: int, task_family: str, difficulty: str) -> dict:
    return {
        "id": f"candidate-{index:03d}",
        "task_family": task_family,
        "buckets": {
            "difficulty_bucket": difficulty,
            "answer_magnitude_bucket": "small",
            "reasoning_length_bucket": "short",
        },
    }


def _profile(task_family: str, difficulty: str, failures: int) -> dict[str, str]:
    return {
        "task_family": task_family,
        "difficulty_bucket": difficulty,
        "answer_magnitude_bucket": "small",
        "reasoning_length_bucket": "short",
        "count": "10",
        "failures": str(failures),
    }


def test_current_selector_is_not_identifiable_beyond_matching_metadata() -> None:
    candidates = [
        *[_candidate(index, "ratio_change", "easy") for index in range(10)],
        *[_candidate(index + 10, "weighted_aggregation", "hard") for index in range(10)],
    ]
    profiles = [
        _profile("ratio_change", "easy", failures=1),
        _profile("weighted_aggregation", "hard", failures=9),
    ]

    summary, stratum_rows, _, _, robustness_rows, rank_rows, id_rename_rows = (
        _MODULE.analyze_selector_identifiability(
        candidates,
        profiles,
        budget=6,
        seeds=[11, 12, 13],
        )
    )

    assert summary["verdict"] == "fail"
    assert summary["identifiability_status"] == "not_identifiable_beyond_matching_metadata"
    assert summary["score_fields_fully_matched_by_primary_baseline"] is True
    assert summary["has_nonrandom_instance_level_signal"] is False
    assert summary["targeted_matched_strata_l1_delta"] == 0
    assert summary["training_allowed"] is False
    assert summary["input_order_invariant"] is True
    assert summary["candidate_id_rename_jaccard"] < 1.0
    assert summary["within_stratum_top_hash_match_rate"] == 1.0
    assert "fixed_quota_random_null_mean_jaccard" in summary
    assert summary["null_pairwise_samples_independent"] is False
    assert all(row["unique_nonrandom_weight_count_within_stratum"] == 1 for row in stratum_rows)
    assert any(row["metric"] == "candidate_id_rename_jaccard" for row in robustness_rows)
    assert all(row["selected_ids_equal_top_hash_k"] for row in rank_rows)
    assert len(id_rename_rows) == len(candidates)


def test_audit_rejects_invalid_budget() -> None:
    candidates = [_candidate(0, "ratio_change", "easy")]
    profiles = [_profile("ratio_change", "easy", failures=1)]

    try:
        _MODULE.analyze_selector_identifiability(
            candidates,
            profiles,
            budget=2,
            seeds=[1],
        )
    except ValueError as exc:
        assert "candidate count" in str(exc)
    else:
        raise AssertionError("expected ValueError")
