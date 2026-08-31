from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load_builder() -> ModuleType:
    scripts_path = str(ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    path = ROOT / "scripts" / "build_rds_public_evidence.py"
    spec = importlib.util.spec_from_file_location("build_rds_public_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load public evidence builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract() -> dict:
    return {
        "scope": "formal_10000_candidate_pool",
        "representation": {"version": "frozen"},
        "thermal": {"abort_temperature_c": 82},
        "selection": {"budget": 500},
        "protocol": {"seed": 20260722},
        "prepared_candidate_scope_count": 10000,
        "prepared_query_scope_count": 448,
        "source_git_commit": "a" * 40,
        "input_bindings": {"data": {"path": "data/file.jsonl", "sha256": "b" * 64}},
        "implementation_bindings": {
            "script": {"path": "scripts/run.py", "sha256": "c" * 64}
        },
        "command": [
            r"C:\Users\person\Python\python.exe",
            "scripts/run.py",
        ],
        "run_contract_sha256": "d" * 64,
        "claim_boundary": "ranking only",
    }


def test_public_contract_replaces_machine_python_without_mutating_source() -> None:
    builder = _load_builder()
    local = _contract()
    public = builder._safe_contract(
        local_contract=local,
        local_contract_file_sha256="e" * 64,
    )
    assert local["command"][0].startswith("C:")
    assert public["command"][0] == "python"
    assert public["source_local_evidence"]["run_contract_self_sha256"] == "d" * 64
    assert public["sanitization"]["immutable_local_evidence_changed"] is False
    text = builder.json.dumps(public, ensure_ascii=False)
    assert builder._scan_text(label="public_contract.json", text=text) == []


def test_public_scan_rejects_paths_secrets_and_raw_text_fields() -> None:
    builder = _load_builder()
    assert builder._scan_text(label="safe", text='{"path":"configs/a.json"}') == []
    findings = builder._scan_text(
        label="unsafe",
        text=(
            '{"exe":"C:\\\\Users\\\\person\\\\python.exe",'
            '"question":"raw",'
            '"api_key":"abcdefghijklmnop"}'
        ),
    )
    finding_types = {row["finding_type"] for row in findings}
    assert finding_types == {
        "windows_absolute_path",
        "secret_like_value",
        "raw_source_text_field",
    }
