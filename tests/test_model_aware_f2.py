from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_model_aware_f2.py"
SCRIPT_DIR = str(SCRIPT_PATH.parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location("run_model_aware_f2", SCRIPT_PATH)
assert SPEC and SPEC.loader
F2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F2)


def _candidate(identifier: str, difficulty: str, value: int) -> dict:
    term_count = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    values = list(range(value, value + term_count))
    weights = [round(1 / term_count, 3)] * term_count
    answer = sum(number * weight for number, weight in zip(values, weights))
    return {
        "id": identifier,
        "split": "candidate_pool",
        "task_family": "weighted_aggregation",
        "difficulty": difficulty,
        "buckets": {
            "difficulty_bucket": difficulty,
            "answer_magnitude_bucket": "small",
            "reasoning_length_bucket": "long",
        },
        "metadata": {"params": {"values": values, "weights": weights}},
        "prompt": f"Weighted values {values} with weights {weights}. What is the aggregate?",
        "rationale": f"Compute the weighted sum = {answer}.",
        "answer": answer,
    }


def test_candidate_sampling_uses_frozen_quotas_and_not_id_or_order() -> None:
    candidates = [
        _candidate(f"candidate-{difficulty}-{index}", difficulty, 10 + index)
        for difficulty in ("easy", "medium", "hard")
        for index in range(10)
    ]
    selected = F2.select_f2_candidates(candidates)
    selected_hashes = {F2.f01.content_hash(row) for row in selected}
    counts = {difficulty: 0 for difficulty in F2.CANDIDATE_QUOTAS}
    for row in selected:
        counts[row["difficulty"]] += 1
    assert counts == F2.CANDIDATE_QUOTAS

    renamed_reversed = [
        {**row, "id": f"renamed-{index}"}
        for index, row in enumerate(reversed(candidates))
    ]
    repeated_hashes = {
        F2.f01.content_hash(row) for row in F2.select_f2_candidates(renamed_reversed)
    }
    assert repeated_hashes == selected_hashes


def test_group_scores_match_prototype_definition() -> None:
    query_count = F2.EXPECTED_ERROR_COUNT + F2.EXPECTED_CORRECT_COUNT
    gram = [[1.0 if left == right else 0.0 for right in range(query_count)] for left in range(query_count)]
    cross = []
    for candidate_index in range(F2.EXPECTED_CANDIDATE_COUNT):
        row = [0.0] * query_count
        row[candidate_index] = 1.0
        row[F2.EXPECTED_ERROR_COUNT + candidate_index] = 0.5
        cross.append(row)
    scores = F2.compute_group_scores(
        cross,
        gram,
        tuple(range(F2.EXPECTED_ERROR_COUNT)),
        tuple(range(F2.EXPECTED_ERROR_COUNT, query_count)),
    )
    assert len(scores["s_e"]) == F2.EXPECTED_CANDIDATE_COUNT
    assert all(value > 0 for value in scores["s_e"])
    assert all(value > 0 for value in scores["s_c"])
    assert all(left > right for left, right in zip(scores["s_e"], scores["s_c"]))


def test_permutation_is_unique_deterministic_and_excludes_observed() -> None:
    query_count = F2.EXPECTED_ERROR_COUNT + F2.EXPECTED_CORRECT_COUNT
    gram = [[1.0 if left == right else 0.0 for right in range(query_count)] for left in range(query_count)]
    cross = [
        [((candidate + 1) * (query + 3) % 17) / 100.0 for query in range(query_count)]
        for candidate in range(F2.EXPECTED_CANDIDATE_COUNT)
    ]
    first = F2.run_permutation_analysis(cross, gram)
    second = F2.run_permutation_analysis(cross, gram)
    assignments = [tuple(item["pseudo_error_indices"]) for item in first["records"]]
    assert first == second
    assert len(assignments) == F2.PERMUTATION_COUNT
    assert len(set(assignments)) == F2.PERMUTATION_COUNT
    assert tuple(range(F2.EXPECTED_ERROR_COUNT)) not in assignments


def test_prompt_only_bm25_normalizes_numeric_values() -> None:
    candidates = [
        {"prompt": "A weighted metric uses weight 0.5 with value 10."},
        {"prompt": "A weighted metric uses weight 0.7 with value 999."},
    ]
    errors = [{"prompt": "A weighted metric uses weight 0.2 with value 42."}]
    scores = F2.bm25_prompt_only_scores(candidates, errors)
    assert scores[0] == scores[1]


def test_empirical_percentile_uses_nearest_rank() -> None:
    assert F2.empirical_percentile(list(range(1, 11)), 0.90) == 9
    assert F2.empirical_percentile(list(range(1, 18)), 0.10) == 2


def test_f2_source_has_no_optimizer_step_or_training_subset_writer() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert ".step(" not in source
    assert "training_subset_written\": False" in source
    assert "f01.serialize_teacher_forcing" in source
    assert "enable_thinking" not in source or '"enable_thinking": False' in source


def test_stage_plan_keeps_next_stage_closed() -> None:
    path = ROOT / "workflow/packets/model_aware_signal_f2_stage_plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["stage_id"] == F2.STAGE_ID
    assert plan["next_stage_if_passed"] is None
