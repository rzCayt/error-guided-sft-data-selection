from eg_sft.experiment.budget_equivalent_phase1_analysis import (
    independent_selection_item_bootstrap,
    require_all_audited,
    threshold_diagnostic,
    validate_task_alignment,
)


def _rows(values):
    return [
        {
            "record_id": f"r{index}",
            "numeric_correct": bool(value),
            "strict_parse_status": "ok",
            "parse_status": "ok",
        }
        for index, value in enumerate(values)
    ]


def _cell(value):
    return {"tasks": {task: _rows([value, value, value]) for task in ("gsm8k", "svamp", "asdiv_numeric", "multiarith")}}


def test_unblinding_gate_requires_all_sixteen_audited() -> None:
    registry = {
        "audited_pass_count": 15,
        "jobs": [{"cell_id": str(i), "status": "AUDITED_PASS"} for i in range(15)],
    }
    try:
        require_all_audited(registry)
    except ValueError as error:
        assert "16/16" in str(error)
    else:
        raise AssertionError("partial registry must not unblind")


def test_alignment_rejects_changed_record_order() -> None:
    left = {"tasks": {task: _rows([1, 0]) for task in ("gsm8k", "svamp", "asdiv_numeric", "multiarith")}}
    right = {"tasks": {task: list(reversed(_rows([1, 0]))) for task in ("gsm8k", "svamp", "asdiv_numeric", "multiarith")}}
    try:
        validate_task_alignment([left, right])
    except ValueError as error:
        assert "membership/order" in str(error)
    else:
        raise AssertionError("changed evaluation order must fail closed")


def test_bootstrap_resamples_selection_lists_independently() -> None:
    report = independent_selection_item_bootstrap(
        left_cells=[_cell(1) for _ in range(4)],
        right_cells=[_cell(0) for _ in range(4)],
        metric="accuracy",
        replicates=10_000,
        seed=7,
    )
    gsm8k = report["tasks"]["gsm8k"]
    assert gsm8k["point_difference"] == 1.0
    assert gsm8k["ci95"] == [1.0, 1.0]
    assert report["selection_resampling"].startswith("independent_within_method")


def test_preregistered_threshold_diagnostics() -> None:
    assert threshold_diagnostic(point=0.02, ci90=[0.01, 0.03], ci95=[0.005, 0.035]) == "supports_practical_gain_threshold"
    assert threshold_diagnostic(point=0.0, ci90=[-0.005, 0.008], ci95=[-0.02, 0.02]) == "supports_approximate_equivalence_band"
    assert threshold_diagnostic(point=-0.02, ci90=[-0.03, -0.01], ci95=[-0.035, -0.005]) == "supports_stable_harm_threshold"
