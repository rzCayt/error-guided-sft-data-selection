from datetime import UTC, datetime

import pytest

from eg_sft.experiment.run_manifest import create_run_manifest, stable_config_hash


def test_config_hash_is_order_invariant() -> None:
    assert stable_config_hash({"a": 1, "b": 2}) == stable_config_hash(
        {"b": 2, "a": 1}
    )


def test_run_directory_cannot_be_overwritten(tmp_path) -> None:
    arguments = {
        "output_root": tmp_path,
        "repo_root": tmp_path,
        "stage": "smoke",
        "config": {"learning_rate": 0.001},
        "seed": 17,
        "command": ["python", "train.py"],
        "dataset_revisions": {"dataset": "revision"},
        "model_revision": "model-revision",
        "now": datetime(2026, 7, 23, 0, 0, tzinfo=UTC),
    }
    run_dir, manifest = create_run_manifest(**arguments)
    assert (run_dir / "manifest.json").is_file()
    assert manifest["config_hash"] == stable_config_hash(arguments["config"])
    assert "git_is_dirty" in manifest
    assert "git_diff_sha256" in manifest

    with pytest.raises(FileExistsError):
        create_run_manifest(**arguments)
