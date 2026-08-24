from __future__ import annotations

import json
import tarfile

import pytest

from eg_sft.experiment.cell_evidence_package import package_cell_evidence


def _write(path, content=b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _complete_run(tmp_path):
    run = tmp_path / "run"
    manifest = {
        "run_id": "run_001",
        "git_commit": "a" * 40,
        "config_hash": "b" * 64,
        "config": {"cell_id": "cell_001"},
    }
    _write(run / "manifest.json", json.dumps(manifest).encode())
    for relative in [
        "resolved_recipe.json",
        "optimizer_step_tokens.jsonl",
        "runtime_events.jsonl",
        "cell_complete.json",
        "audit/formal_cell_audit.json",
        "audit/formal_cell_audit.sha256",
        "audit/ood_audit.json",
        "audit/ood_audit.sha256",
        "training_complete/training_metrics.json",
        "training_complete/token_audit.json",
        "training_complete/token_budget_audit.json",
        "training_complete/adapter/adapter_model.safetensors",
        "training_complete/adapter/adapter_config.json",
        "evaluation/merged/raw_outputs.jsonl",
        "checkpoints/checkpoint_step_008.json",
        "checkpoints/checkpoint_step_008.pt",
    ]:
        _write(run / relative)
    return run


def test_package_is_compact_manifested_and_excludes_checkpoint_tensors(tmp_path) -> None:
    run = _complete_run(tmp_path)
    output = tmp_path / "cell.tar.gz"
    manifest, archive_sha = package_cell_evidence(run_dir=run, output=output)
    assert len(archive_sha) == 64
    assert manifest["accuracy_withheld"] is True
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "EVIDENCE_MANIFEST.json" in names
    assert "checkpoints/checkpoint_step_008.json" in names
    assert "checkpoints/checkpoint_step_008.pt" not in names


def test_package_refuses_missing_audit_and_overwrite(tmp_path) -> None:
    run = _complete_run(tmp_path)
    (run / "audit" / "ood_audit.json").rename(tmp_path / "ood_audit.json.moved")
    with pytest.raises(ValueError, match="missing required evidence"):
        package_cell_evidence(run_dir=run, output=tmp_path / "cell.tar.gz")

    _write(run / "audit" / "ood_audit.json")
    output = tmp_path / "cell.tar.gz"
    package_cell_evidence(run_dir=run, output=output)
    with pytest.raises(FileExistsError):
        package_cell_evidence(run_dir=run, output=output)
