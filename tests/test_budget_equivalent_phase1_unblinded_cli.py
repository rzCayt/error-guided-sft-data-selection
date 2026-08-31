import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_budget_equivalent_phase1_unblinded.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("phase1_unblinded_cli", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_claim_boundary_explicitly_blocks_confirmatory_claims() -> None:
    methods = {
        "method": {
            "selection_replicate_count": 4,
            "tasks": {
                "gsm8k": {"accuracy": {"mean": 0.5, "selection_replicate_sd": 0.1}},
                "ood_macro": {"accuracy": {"mean": 0.4, "selection_replicate_sd": 0.1}},
            },
        }
    }
    comparisons = {
        "comparisons": {
            "comparison": {
                "accuracy": {
                    "tasks": {
                        "gsm8k": {
                            "point_difference": 0.01,
                            "ci95": [-0.01, 0.03],
                        }
                    }
                },
                "primary_gsm8k_threshold_diagnostic": "insufficient_evidence",
            }
        }
    }
    text = MODULE._claim_boundary_markdown(
        method_summary=methods, comparisons=comparisons
    )
    assert "不允许" in text
    assert "一个训练随机种子" in text
    assert "最终有效" in text
