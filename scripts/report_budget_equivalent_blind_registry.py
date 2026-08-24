"""Report Phase 1 progress without exposing method names or accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_blind import blinded_registry  # noqa: E402
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    phase1_registry,
    read_json_object,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = blinded_registry(
        private_map=read_json_object(args.private_map.resolve()),
        registry=phase1_registry(
            repo_root=ROOT,
            config_path=args.config.resolve(),
        ),
    )
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
