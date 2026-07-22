import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_public_release_artifacts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_public_release_artifacts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_private_source(tmp_path: Path) -> Path:
    """Create a self-contained source fixture with machine-specific model paths."""
    source = tmp_path / "private_source"
    pipeline = source / "qwen3_1_7b_exact_chat_pipeline_check_25"
    selector = source / "selector_identifiability_rerun"
    shutil.copytree(ROOT / "results" / "public_release_v1" / "model_pipeline_check_25", pipeline)
    shutil.copytree(
        ROOT / "results" / "public_release_v1" / "selector_identifiability_rerun", selector
    )

    jsonl = pipeline / "scale_model_diagnostic_outputs.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["model"] = r"E:\private-cache\Qwen3-1.7B"
    jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    metadata_path = pipeline / "scale_model_diagnostic_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model"] = r"E:\private-cache\Qwen3-1.7B"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    summary_path = pipeline / "scale_model_diagnostic_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
        fieldnames = list(summary_rows[0])
    summary_rows[0]["model"] = r"E:\private-cache\Qwen3-1.7B"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    return source


def test_build_bundle_sanitizes_model_paths_and_emits_manifest(tmp_path):
    module = load_module()
    source = make_private_source(tmp_path)
    output = tmp_path / "public_release"

    generated = module.build_bundle(source, output)

    assert len(generated) >= 10
    jsonl = output / "model_pipeline_check_25" / "scale_model_diagnostic_outputs.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 25
    assert {row["model"] for row in rows} == {"Qwen/Qwen3-1.7B"}

    metadata = json.loads(
        (output / "model_pipeline_check_25" / "scale_model_diagnostic_run_metadata.json")
        .read_text(encoding="utf-8")
    )
    assert metadata["model"] == "Qwen/Qwen3-1.7B"
    assert metadata["model_revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

    with (output / "model_pipeline_check_25" / "scale_model_diagnostic_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        summary = next(csv.DictReader(handle))
    assert summary["model"] == "Qwen/Qwen3-1.7B"
    assert float(summary["numeric_accuracy"]) == pytest.approx(0.76)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_build_bundle_refuses_to_overwrite(tmp_path):
    module = load_module()
    source = ROOT / "results" / "professor_package_validation"
    output = tmp_path / "public_release"
    output.mkdir()

    with pytest.raises(FileExistsError):
        module.build_bundle(source, output)
