import importlib.util
from pathlib import Path

from eg_sft.data.generator import example_signature, generate_all, generate_split
from eg_sft.selection.bias_audit import audit_selection_bias, summarize_selection_bias
from eg_sft.selection.error_guided import select_error_guided
from eg_sft.selection.matched_random import select_matched_random
from eg_sft.selection.strong_baselines import (
    select_exact_matched_random_multi_seed,
    select_metadata_hard_baseline,
    select_stratified_random,
    summarize_baseline_suite,
)

_AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_real_diagnostic_errors.py"
_AUDIT_SPEC = importlib.util.spec_from_file_location("audit_real_diagnostic_errors", _AUDIT_SCRIPT)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
audit_category = _AUDIT_MODULE.audit_category
numeric_tokens = _AUDIT_MODULE.numeric_tokens


def test_generator_is_deterministic() -> None:
    left = [row.to_dict() for row in generate_split("candidate_pool", n=8, seed=123)]
    right = [row.to_dict() for row in generate_split("candidate_pool", n=8, seed=123)]
    assert left == right
    assert {row["task_family"] for row in left}


def test_generate_all_has_unique_cross_split_signatures() -> None:
    generated = generate_all(seed=123)
    signatures = [
        example_signature(row.to_dict())
        for examples in generated.values()
        for row in examples
    ]
    assert len(signatures) == len(set(signatures))


def test_selection_budget_and_matching() -> None:
    candidates = [row.to_dict() for row in generate_split("candidate_pool", n=80, seed=123)]
    profile = [
        {
            "task_family": "temporal_numeric_constraint",
            "difficulty_bucket": "hard",
            "answer_magnitude_bucket": "medium",
            "reasoning_length_bucket": "long",
            "count": "10",
            "failures": "8",
            "error_rate": "0.8",
        }
    ]
    targeted = select_error_guided(candidates, profile, budget=16, seed=1)
    matched = select_matched_random(candidates, targeted, seed=2)
    assert len(targeted) == 16
    assert len(matched) == 16
    assert {row["id"] for row in targeted}.isdisjoint({row["id"] for row in matched})
    audit = audit_selection_bias(targeted, matched)
    assert all(row["count_delta"] == 0 for row in audit)
    summary = summarize_selection_bias(targeted, matched)
    metrics = {row["metric"]: row for row in summary}
    assert "overlap_rate" in metrics
    assert "mean_abs_answer" in metrics


def test_strong_baseline_suite_has_budget_and_audit_rows() -> None:
    candidates = [row.to_dict() for row in generate_split("candidate_pool", n=120, seed=456)]
    profile = [
        {
            "task_family": "temporal_numeric_constraint",
            "difficulty_bucket": "hard",
            "answer_magnitude_bucket": "medium",
            "reasoning_length_bucket": "long",
            "count": "10",
            "failures": "9",
            "error_rate": "0.9",
        }
    ]
    targeted = select_error_guided(candidates, profile, budget=24, seed=3)
    matched_many = select_exact_matched_random_multi_seed(candidates, targeted, seeds=[11, 12, 13])
    stratified = select_stratified_random(candidates, budget=24, seed=14)
    hard = select_metadata_hard_baseline(candidates, profile, budget=24, seed=15)
    baselines = {
        **{f"exact_matched_random_seed_{seed}": rows for seed, rows in matched_many.items()},
        "stratified_random": stratified,
        "metadata_hard_baseline": hard,
    }

    assert all(len(rows) == 24 for rows in baselines.values())
    assert set(matched_many) == {11, 12, 13}
    summary = summarize_baseline_suite(targeted, baselines)
    summary_baselines = {row["baseline"] for row in summary}
    assert set(baselines).issubset(summary_baselines)
    assert "exact_matched_random_multi_seed" in summary_baselines
    assert "strata_l1_delta" in {row["metric"] for row in summary}
    assert any(str(row["metric"]).endswith("_variance") for row in summary)


def test_real_diagnostic_error_audit_categories() -> None:
    multi_number_wrong = {
        "numeric_accuracy": False,
        "parse_success": True,
        "task_family": "ratio_change",
        "raw_continuation": " 75 - 5% of 75 = 75 - 3",
    }
    weighted_wrong = {
        "numeric_accuracy": False,
        "parse_success": True,
        "task_family": "weighted_aggregation",
        "raw_continuation": " 150.00000000000001",
    }
    direct_wrong = {
        "numeric_accuracy": False,
        "parse_success": True,
        "task_family": "multiplicative_relation",
        "raw_continuation": " 1260",
    }

    assert len(numeric_tokens(multi_number_wrong["raw_continuation"])) >= 3
    assert audit_category(multi_number_wrong) == "parser_or_output_format_risk"
    assert audit_category(weighted_wrong) == "prompt_or_task_misunderstanding_risk"
    assert audit_category(direct_wrong) == "model_calculation_or_reasoning_error"
