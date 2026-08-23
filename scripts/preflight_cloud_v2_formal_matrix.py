"""Print the frozen nine-cell registry without launching training or evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.cloud_v2_formal import build_formal_registry  # noqa: E402


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
        "--config",
        type=Path,
        default=Path("configs/cloud_v2_formal_b500_single_cell_fixed_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_formal_registry(
        repo_root=ROOT,
        config_path=args.config.resolve(),
        python_executable=sys.executable,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
