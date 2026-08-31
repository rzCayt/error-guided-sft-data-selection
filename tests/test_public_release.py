from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_summary_recomputes_exactly() -> None:
    module = load_script("reproduce_public_summary.py")
    assert module.build_payload() == module.read_json(module.OUTPUTS["json"])


def test_public_release_core_checks() -> None:
    module = load_script("verify_public_release.py")
    summary = module.json.loads(module.ROOT.joinpath("results/public_summary/main_results.json").read_text(encoding="utf-8"))
    module.check_required_paths()
    module.check_evidence_hashes(summary)
    module.check_readme_results(summary)
    module.check_historical_banners()
    module.check_markdown_links()
    module.check_generated_manifests()
