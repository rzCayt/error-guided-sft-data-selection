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
clean_problem_prompt = _AUDIT_MODULE.clean_problem_prompt
translate_problem_prompt = _AUDIT_MODULE.translate_problem_prompt
numeric_tokens = _AUDIT_MODULE.numeric_tokens
write_csv_for_spreadsheets = _AUDIT_MODULE.write_csv_for_spreadsheets
write_titled_csv_for_spreadsheets = _AUDIT_MODULE.write_titled_csv_for_spreadsheets


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
        "answer": 71.25,
        "parsed_prediction": 3.0,
        "raw_continuation": " 75 - 5% of 75 = 75 - 3",
    }
    parser_tail_wrong = {
        "numeric_accuracy": False,
        "parse_success": True,
        "task_family": "ratio_change",
        "answer": 48.0,
        "parsed_prediction": 4.0,
        "raw_continuation": " 60 - 20% of 60 = 60 - 12 = 48 | What is the answer? 4",
    }
    weighted_wrong = {
        "numeric_accuracy": False,
        "parse_success": True,
        "task_family": "weighted_aggregation",
        "answer": 72.87,
        "parsed_prediction": 150.0,
        "raw_continuation": " 150.00000000000001",
    }
    direct_wrong = {
        "numeric_accuracy": False,
        "parse_success": True,
        "task_family": "multiplicative_relation",
        "answer": 15120.0,
        "parsed_prediction": 1260.0,
        "raw_continuation": " 1260",
    }

    assert len(numeric_tokens(multi_number_wrong["raw_continuation"])) >= 3
    assert audit_category(multi_number_wrong) == "percent_change_calculation_error"
    assert audit_category(parser_tail_wrong) == "parser_tail_fragment_risk"
    assert audit_category(weighted_wrong) == "weighted_formula_error"
    assert audit_category(direct_wrong) == "multiplication_calculation_error"


def test_real_diagnostic_audit_csv_is_excel_friendly_utf8(tmp_path) -> None:
    path = tmp_path / "audit.csv"
    write_csv_for_spreadsheets(path, [{"human_check_prompt": "请判断这是真推理错误"}])

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert "请判断这是真推理错误" in path.read_text(encoding="utf-8-sig")


def test_chinese_review_csv_has_title_and_chinese_headers(tmp_path) -> None:
    path = tmp_path / "audit_cn.csv"
    write_titled_csv_for_spreadsheets(
        path,
        "真实错误画像人工复核表：Qwen2.5-0.5B base diagnostic",
        [{"样例编号": "dev_diagnostic-0000", "任务类型": "比例变化"}],
    )

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert lines[0] == "真实错误画像人工复核表：Qwen2.5-0.5B base diagnostic"
    assert lines[2].startswith("样例编号,任务类型")


def test_problem_prompt_translation_for_review_table() -> None:
    prompt = (
        "Problem: A metric starts at 36 and then has a 15% decrease. "
        "What is the final value?\nFinal numeric answer ="
    )

    assert clean_problem_prompt(prompt) == (
        "A metric starts at 36 and then has a 15% decrease. What is the final value?"
    )
    assert translate_problem_prompt(prompt) == "一个指标从 36 开始，随后下降 15%。最终值是多少？"
