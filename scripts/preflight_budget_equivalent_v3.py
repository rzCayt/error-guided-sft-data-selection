"""Run the CPU-only, accuracy-blind v3 protocol preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_protocol import preflight_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/budget_equivalent_v3_protocol.json"),
    )
    args = parser.parse_args()
    report = preflight_protocol(repo_root=ROOT, config_path=args.config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
