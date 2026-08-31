from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_progress_entry_is_cpu_only_and_does_not_emit_accuracy() -> None:
    source = (ROOT / "scripts" / "aggregate_phase2_v7_progress.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = "\n".join(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    assert "torch" not in imports
    assert "transformers" not in imports
    assert '"accuracy_withheld": True' in source
    assert '"automatic_unblinding": False' in source
