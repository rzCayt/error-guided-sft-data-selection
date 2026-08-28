from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v8_release_bundle_includes_source_and_only_canonical_configs() -> None:
    source = (ROOT / "scripts/build_phase2_v8_release_bundle.py").read_text(
        encoding="utf-8"
    )
    assert "independent_review_with_actual_source" in source
    assert "CANONICAL_RUNTIME_FILES_v8_RELEASE.json" in source
    assert "phase2_clean_common24_v8_canonical.json" in source
    assert 'ROOT / "configs"' in source
    assert 'ROOT / "configs",' in source
    assert "test-only/noncanonical" in (
        ROOT / "docs/20260828_PHASE2_V8_RELEASE_README.md"
    ).read_text(encoding="utf-8")
    assert "phase2_v8_materialized_contracts_v4" in source
    assert "SSH_ENDPOINT" in source
    assert "selected = list(files)" in source
