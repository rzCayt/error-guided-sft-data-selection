from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_materialized_contract_audit_entry_is_read_only_and_seed_aware() -> None:
    source = (ROOT / "scripts/audit_phase2_v8_materialized_contracts.py").read_text(
        encoding="utf-8"
    )
    assert "SEED_INVARIANT_FIELDS" in source
    assert "SEED_DERIVED_FIELDS" in source
    assert '"three_expected_seeds"' in source
    assert '"tokenizer_is_qwen2"' in source
    assert "write_exclusive_or_verify" in source
    assert "AutoTokenizer" not in source
