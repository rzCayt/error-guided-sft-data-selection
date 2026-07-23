import importlib.util
import sys
from pathlib import Path


_PATH = Path(__file__).parents[1] / "scripts" / "run_rds_h1a_scoring.py"
sys.path.insert(0, str(_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_rds_h1a_scoring", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_complete_rank_spearman_and_top_jaccard() -> None:
    assert _MODULE._spearman_from_complete_ranks([0, 1, 2], [0, 1, 2]) == 1.0
    assert _MODULE._spearman_from_complete_ranks([0, 1, 2], [2, 1, 0]) == -1.0
    assert _MODULE._top_jaccard([0, 1, 2], [1, 2, 0], 2) == 1 / 3


def test_reliability_candidates_span_the_complete_error_order() -> None:
    order = list(range(96))
    selected = _MODULE._reliability_indices(order, count=10)
    assert selected[0] == 0
    assert selected[-1] == 95
    assert len(selected) == len(set(selected)) == 10
