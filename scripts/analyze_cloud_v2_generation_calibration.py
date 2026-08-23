"""Compare four 128-row cloud-v2 batched-generation calibration runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.cloud_v2_analysis import (  # noqa: E402
    analyze_generation_calibration,
    parse_named_paths,
    read_json_object,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _input_contract(run_paths: dict[int, Path]) -> dict:
    fields = (
        "calibration_config_hash",
        "adapter_sha256",
        "source_training_run_id",
        "source_training_config_hash",
        "protocol_config_sha256",
        "generation_protocol_split",
        "generation_example_count",
    )
    manifests = {
        batch_size: read_json_object(path / "manifest.json")
        for batch_size, path in run_paths.items()
    }
    metrics = {
        batch_size: read_json_object(path / "metrics.json")
        for batch_size, path in run_paths.items()
    }
    reference = manifests[1]["config"]
    field_checks = {
        field: all(
            manifest["config"].get(field) == reference.get(field)
            for manifest in manifests.values()
        )
        for field in fields
    }
    metric_status_checks = {
        str(batch_size): payload.get("status") == "PASS"
        for batch_size, payload in metrics.items()
    }
    return {
        "field_checks": field_checks,
        "metric_status_checks": metric_status_checks,
        "pass": all(field_checks.values()) and all(metric_status_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="BATCH=PATH",
        help="Repeat exactly once for b1, b4, b8, and b16.",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("configs/cloud_v2_calibration_analysis_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.analysis_config.resolve()
    config = read_json_object(config_path)
    if config.get("analysis_version") != "cloud-v2-calibration-analysis-v1":
        raise ValueError("unexpected cloud-v2 analysis config version")
    named = parse_named_paths(args.run, expected_names=("b1", "b4", "b8", "b16"))
    run_paths = {int(name[1:]): path for name, path in named.items()}
    generation_config = config["generation"]
    report = analyze_generation_calibration(
        run_paths=run_paths,
        expected_count=int(generation_config["expected_example_count"]),
        max_difference_examples=int(generation_config["max_difference_examples"]),
    )
    contract = _input_contract(run_paths)
    report["input_contract"] = contract
    if not contract["pass"]:
        report["status"] = "FAIL"
    report["analysis_config_sha256"] = file_sha256(config_path)
    report["claim_boundary"] = (
        "This report checks batch-generation equivalence on the development split only. "
        "It is not a held-out model-quality result."
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
