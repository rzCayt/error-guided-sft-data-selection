import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_model_native_baseline_table.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("build_model_native_baseline_table", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_boxed_answer_counts_as_math_native_auxiliary_not_strict() -> None:
    row = {
        "answer": 72.87,
        "raw_continuation": "Therefore, the final answer is: \\[ \\boxed{72.87} \\]",
        "numeric_accuracy": True,
    }

    assert _MODULE.row_boxed_correct(row) is True
    assert _MODULE.row_math_native_aux_correct(row) is True
    assert _MODULE.row_strict_correct(row) is False


def test_strict_final_answer_counts_as_strict() -> None:
    row = {
        "answer": 72.87,
        "raw_continuation": "Final answer: 72.87",
        "numeric_accuracy": True,
    }

    assert _MODULE.row_strict_correct(row) is True
    assert _MODULE.row_math_native_aux_correct(row) is False


def test_formula_gold_counts_as_math_native_auxiliary() -> None:
    row = {
        "answer": 43.326,
        "raw_continuation": "21.978 + 9.324 + 12.024 = 43.326",
        "numeric_accuracy": True,
    }

    assert _MODULE.row_math_native_aux_correct(row) is True
    assert _MODULE.row_strict_correct(row) is False
