"""Build the four Phase 1 selections for every frozen selection replicate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_lists import build_phase1_lists  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/budget_equivalent_v3_protocol.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--engineering-allow-exact-prompt-fallback",
        action="store_true",
        help="Build non-formal engineering lists when fuzzy duplicate clusters are absent.",
    )
    args = parser.parse_args()
    result = build_phase1_lists(
        repo_root=ROOT,
        config_path=args.config.resolve(),
        output_root=args.output_root,
        engineering_allow_exact_prompt_fallback=args.engineering_allow_exact_prompt_fallback,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
