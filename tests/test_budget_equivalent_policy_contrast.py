from eg_sft.experiment.budget_equivalent_policy_contrast import (
    common_stratum_contrast,
    pair_policy_contrast,
    source_diversity,
)


def _row(candidate_id, source, response, total, stratum="s"):
    return {
        "candidate_id": candidate_id,
        "source_dataset": source,
        "supervised_tokens": response,
        "total_tokens": total,
        "common_mix_stratum": stratum,
    }


def test_source_diversity_reports_effective_count() -> None:
    report = source_diversity(
        [_row("a", "x", 1, 3), _row("b", "y", 1, 3)]
    )
    assert report["source_counts"] == {"x": 1, "y": 1}
    assert abs(report["effective_source_count"] - 2.0) < 1e-12


def test_pair_contrast_uses_same_rds_score_for_both_policies() -> None:
    rds = [_row("a", "x", 2, 5), _row("b", "x", 2, 5)]
    random = [_row("c", "x", 2, 5), _row("d", "x", 2, 5)]
    scores = {"a": 1.0, "b": 0.9, "c": 0.2, "d": 0.1}
    report = pair_policy_contrast(
        rds_rows=rds, random_rows=random, rds_priority_by_id=scores
    )
    assert report["selected_id_jaccard"] == 0.0
    assert report["replacement_fraction_of_budget"] == 1.0
    assert abs(report["mean_rds_rank_percentile_lift"] - 0.8) < 1e-12
    assert report["rds_score_direction_is_positive"] is True


def test_common_stratum_contrast_requires_matching_quotas() -> None:
    rds = [_row("a", "x", 2, 5), _row("b", "x", 2, 5)]
    random = [_row("c", "x", 2, 5)]
    try:
        common_stratum_contrast(
            rds_rows=rds,
            random_rows=random,
            rds_priority_by_id={"a": 1.0, "b": 0.9, "c": 0.2},
            stratum_candidate_counts={"s": 10},
        )
    except ValueError as error:
        assert "quotas differ" in str(error)
    else:
        raise AssertionError("mismatched common quotas must fail closed")
