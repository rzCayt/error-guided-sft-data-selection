"""Compare four cloud-v2 training calibration runs without using a GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.cloud_v2_analysis import (  # noqa: E402
    TRAINING_PROFILES,
    analyze_training_calibration,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="PROFILE=PATH",
        help="Repeat exactly once for mb1_ga16, mb2_ga8, mb4_ga4, and mb8_ga2.",
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
    run_paths = parse_named_paths(args.run, expected_names=TRAINING_PROFILES)
    report = analyze_training_calibration(
        run_paths=run_paths,
        thresholds=config["training"],
    )
    report["analysis_config_sha256"] = file_sha256(config_path)
    report["claim_boundary"] = (
        "This report calibrates execution equivalence and throughput. It cannot establish "
        "random, rds_all, or rds_error downstream effectiveness."
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
