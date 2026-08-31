from __future__ import annotations

from pathlib import Path

import pytest

from eg_sft.experiment.phase2_v8_canonical_runtime import validate_canonical_runtime


ROOT = Path(__file__).resolve().parents[1]


def test_only_release_canonical_manifest_is_accepted() -> None:
    release = ROOT / "configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json"
    report = validate_canonical_runtime(repo_root=ROOT, manifest_path=release)
    assert report["manifest"]["status"] == "FROZEN"
    with pytest.raises(ValueError, match="non-release canonical authority"):
        validate_canonical_runtime(
            repo_root=ROOT,
            manifest_path=ROOT / "configs/CANONICAL_RUNTIME_FILES_v8_FINAL.json",
        )
