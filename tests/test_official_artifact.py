import pytest

from eg_sft.artifact.official_tis import summarize_rows


def test_summarize_rows_uses_seed_level_values() -> None:
    rows = [
        {"method": "Random", "true_metric": 80.0},
        {"method": "Random", "true_metric": 82.0},
        {"method": "Random", "true_metric": 84.0},
    ]
    summary = summarize_rows(rows)
    assert summary["Random"]["mean"] == pytest.approx(82.0)
    assert summary["Random"]["sample_std"] == pytest.approx(2.0)
