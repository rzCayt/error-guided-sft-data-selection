from __future__ import annotations

import json
from pathlib import Path

import pytest

from eg_sft.experiment.phase2_v8_snapshot import (
    file_subset_manifest,
    validate_snapshot_manifest,
)


def test_snapshot_manifest_fails_closed_on_file_change(tmp_path: Path) -> None:
    snapshot = tmp_path / "8faed761d45a263340a0528343f099c05c9a4323"
    snapshot.mkdir()
    target = snapshot / "config.json"
    target.write_text("{}", encoding="utf-8")
    manifest = file_subset_manifest(snapshot)
    assert len(validate_snapshot_manifest(snapshot=snapshot, manifest=manifest)) == 64
    target.write_text(json.dumps({"changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot file changed"):
        validate_snapshot_manifest(snapshot=snapshot, manifest=manifest)
