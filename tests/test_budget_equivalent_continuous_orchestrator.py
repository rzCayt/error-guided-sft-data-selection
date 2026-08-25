import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_budget_equivalent_phase1_continuous.py"
SPEC = importlib.util.spec_from_file_location("continuous_phase1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_find_run_dir_rejects_duplicate_cell_runs(tmp_path: Path) -> None:
    for name in ("run_a", "run_b"):
        _write_json(tmp_path / name / "manifest.json", {"config": {"cell_id": "cell"}})
    try:
        MODULE._find_run_dir(tmp_path, "cell")
    except ValueError as error:
        assert "multiple formal run directories" in str(error)
    else:
        raise AssertionError("duplicate formal runs must fail closed")


def test_verified_audit_requires_matching_sha_and_sealed_status(tmp_path: Path) -> None:
    artifact = tmp_path / "audit" / "formal_cell_audit.json"
    _write_json(
        artifact,
        {"status": "PASS", "cell_id": "cell", "accuracy_withheld": True},
    )
    digest = MODULE._sha256(artifact)
    artifact.with_suffix(".sha256").write_text(
        f"{digest}  {artifact.name}\n", encoding="ascii"
    )
    assert MODULE._verified_audit(tmp_path, artifact.name, "cell") is True


def test_verified_audit_rejects_exposed_accuracy_flag(tmp_path: Path) -> None:
    artifact = tmp_path / "audit" / "ood_audit.json"
    _write_json(artifact, {"status": "PASS", "accuracy_withheld": False})
    digest = MODULE._sha256(artifact)
    artifact.with_suffix(".sha256").write_text(
        f"{digest}  {artifact.name}\n", encoding="ascii"
    )
    try:
        MODULE._verified_audit(tmp_path, artifact.name, "cell")
    except ValueError as error:
        assert "invalid sealed audit" in str(error)
    else:
        raise AssertionError("unsealed audit must fail closed")


def test_hash_verifier_accepts_tar_gz_sha_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "cell_evidence.tar.gz"
    artifact.write_bytes(b"evidence")
    digest = MODULE._sha256(artifact)
    artifact.with_suffix(".gz.sha256").write_text(
        f"{digest}  {artifact.name}\n", encoding="ascii"
    )
    MODULE._verify_hash_sidecar(artifact)
