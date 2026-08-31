from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_registry_matches_frozen_matrix() -> None:
    module = load_module(ROOT / "scripts" / "generate_experiment_registry.py")
    assert module.OUTPUT.read_text(encoding="utf-8") == module.build_csv()


def test_frozen_public_config_snapshots_match_runtime_sources() -> None:
    module = load_module(ROOT / "scripts" / "snapshot_public_configs.py")
    module.check_snapshots()


def test_public_figures_match_canonical_result_source() -> None:
    module = load_module(ROOT / "figures" / "generate_public_figures.py")
    module.check_figures()


def test_internal_integrity_audit_is_explicitly_non_independent() -> None:
    module = load_module(ROOT / "scripts" / "verify_public_release.py")
    module.check_internal_audit_disclosure()


def test_public_tree_has_no_sensitive_or_restricted_artifacts() -> None:
    module = load_module(ROOT / "scripts" / "verify_public_release.py")
    files = module.iter_repository_files()
    module.check_sensitive_content(files)
    module.check_public_absolute_paths(files)
    module.check_restricted_artifacts(files)
