from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_worker_entry_keeps_heavy_orchestration_imports_inside_main() -> None:
    path = ROOT / "scripts" / "run_phase2_v7_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top_level_imports.append(ast.unparse(node))
    joined = "\n".join(top_level_imports)
    assert "torch" not in joined
    assert "peft" not in joined
    assert "transformers" not in joined


def test_worker_has_exact_operator_confirmation() -> None:
    source = (ROOT / "scripts" / "run_phase2_v7_worker.py").read_text(
        encoding="utf-8"
    )
    assert "PHASE2_V7_32CELL_BLOCK_APPROVED" in source
    assert "automatic_unblinding\": False" in source
