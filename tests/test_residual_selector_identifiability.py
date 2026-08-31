import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_residual_selector_identifiability.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("audit_residual_selector_identifiability", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(identifier: str, split: str, pct: int, correct: bool | None = None) -> dict:
    row = {
        "id": identifier,
        "split": split,
        "task_family": "ratio_change",
        "prompt": f"A metric starts at 100 and then has a {pct}% increase. What is the final value?",
        "metadata": {"params": {"base": 100, "direction": "increase", "pct": pct}},
        "buckets": {
            "difficulty_bucket": "easy",
            "answer_magnitude_bucket": "medium",
            "reasoning_length_bucket": "short",
        },
    }
    if correct is not None:
        row["numeric_accuracy"] = correct
    return row


def test_operation_signal_is_id_and_order_invariant() -> None:
    candidates = [_row(f"candidate-{index}", "candidate_pool", pct) for index, pct in enumerate([5, 10, 15, 30, 40, 7])]
    diagnostics = [
        _row("dev-1", "dev_diagnostic", 5, False),
        _row("dev-2", "dev_diagnostic", 10, False),
        _row("dev-3", "dev_diagnostic", 30, True),
        _row("dev-4", "dev_diagnostic", 40, True),
    ]

    summary, score_rows, _, invariance, _ = _MODULE.analyze(candidates, diagnostics, budget=3)

    assert summary["input_order_invariant"] is True
    assert summary["candidate_id_rename_invariant"] is True
    assert summary["candidate_answer_or_rationale_used"] is False
    assert summary["training_allowed"] is False
    assert len({row["main_residual_score"] for row in score_rows}) > 1
    checks = {row["check"]: row["passed"] for row in invariance}
    assert checks["input_order_invariance"] is True
    assert checks["candidate_id_rename_invariance"] is True


def test_rejects_test_split() -> None:
    candidates = [_row("candidate-1", "candidate_pool", 5)]
    diagnostics = [_row("test-1", "test_id", 5, False)]

    try:
        _MODULE.analyze(candidates, diagnostics, budget=1)
    except ValueError as exc:
        assert "dev_diagnostic" in str(exc)
    else:
        raise AssertionError("expected test split rejection")


def test_weighted_family_without_correct_contrast_is_ineligible() -> None:
    candidate = {
        "id": "candidate-weighted",
        "split": "candidate_pool",
        "task_family": "weighted_aggregation",
        "prompt": "A weighted metric uses weight 0.5 with value 10, weight 0.5 with value 20.",
        "metadata": {"params": {"values": [10, 20], "weights": [0.5, 0.5]}},
        "buckets": {
            "difficulty_bucket": "easy",
            "answer_magnitude_bucket": "small",
            "reasoning_length_bucket": "long",
        },
    }
    diagnostic = {**candidate, "id": "dev-weighted", "split": "dev_diagnostic", "numeric_accuracy": False}

    scores, eligible = _MODULE.contrastive_scores([candidate], [diagnostic], _MODULE.operation_features)

    assert eligible["candidate-weighted"] is False
    assert scores["candidate-weighted"] == 0.0


def test_candidate_answer_and_rationale_do_not_change_signal() -> None:
    candidates = [_row("candidate-1", "candidate_pool", 5), _row("candidate-2", "candidate_pool", 30)]
    diagnostics = [
        _row("dev-1", "dev_diagnostic", 5, False),
        _row("dev-2", "dev_diagnostic", 30, True),
    ]
    changed = [
        {**row, "answer": 999999.0, "rationale": "deliberately changed"}
        for row in candidates
    ]

    original, _ = _MODULE.contrastive_scores(candidates, diagnostics, _MODULE.operation_features)
    mutated, _ = _MODULE.contrastive_scores(changed, diagnostics, _MODULE.operation_features)

    assert original == mutated
