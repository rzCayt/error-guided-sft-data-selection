import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_public_release.py"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_public_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_release_claims_match_artifacts():
    module = load_module()
    report = module.validate(ROOT / "results" / "public_release_v1")

    assert report["overall_passed"] is True
    assert {check["id"] for check in report["checks"]} == {
        "manifest_hashes",
        "metadata_selector_claim",
        "residual_selector_claim",
        "model_aware_f2_claim",
        "model_pipeline_claim",
    }
