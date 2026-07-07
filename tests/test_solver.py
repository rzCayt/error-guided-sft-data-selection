from eg_sft.data.solver import solve


def test_ratio_change_solver() -> None:
    answer, _ = solve("ratio_change", {"base": 100, "pct": 15, "direction": "increase"})
    assert answer == 115


def test_weighted_aggregation_solver() -> None:
    answer, _ = solve("weighted_aggregation", {"weights": [0.25, 0.75], "values": [80, 100]})
    assert answer == 95


def test_temporal_solver_with_bounds() -> None:
    answer, _ = solve(
        "temporal_numeric_constraint",
        {"start": 50, "deltas": [-100, 40], "floor": 0, "cap": 100},
    )
    assert answer == 40
