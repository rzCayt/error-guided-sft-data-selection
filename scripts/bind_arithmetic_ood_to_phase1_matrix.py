"""Create a new immutable Phase 1 matrix config bound to OOD manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    read_json_object,
    validate_matrix_config,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--ood-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base_matrix.resolve()
    ood_root = args.ood_root.resolve()
    config = read_json_object(base_path)
    validate_matrix_config(config)
    manifest_path = ood_root / "ood_manifest.json"
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != "budget-equivalent-arithmetic-ood-manifest-v1":
        raise ValueError("unexpected OOD manifest version")
    datasets = {}
    for name in ("svamp", "asdiv_numeric", "multiarith"):
        binding = manifest["artifacts"][name]
        path = ood_root / str(binding["path"])
        if file_sha256(path) != binding["sha256"]:
            raise ValueError(f"OOD artifact hash changed: {name}")
        datasets[name] = {
            "path": _relative(path),
            "sha256": binding["sha256"],
            "expected_record_count": int(binding["retained_count"]),
        }
    config["parent_matrix_config"] = {
        "path": _relative(base_path),
        "sha256": file_sha256(base_path),
    }
    config["ood_evaluation"] = {
        "manifest": {
            "path": _relative(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "datasets": datasets,
        "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
        "parser_policy": "strict_final_marker_then_last_numeric_fallback",
        "aggregation": "equal_weight_macro_average_across_three_datasets",
        "required_before_unblinding": True,
        "raw_dataset_text_in_manifest": False,
    }
    config["evaluation"]["secondary_datasets"] = list(datasets)
    config["execution_policy"]["ood_audits_required_before_unblinding"] = True
    output = args.output_config.resolve()
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(file_sha256(output))


if __name__ == "__main__":
    main()
