from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.artifact.official_tis import (  # noqa: E402
    reproduce_qwen3_gsm8k_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the pinned official Qwen3/GSM8K plot artifact."
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=500)
    args = parser.parse_args()

    summary = reproduce_qwen3_gsm8k_artifact(
        official_repo=args.official_repo.resolve(),
        output_dir=args.output_dir.resolve(),
        budget=args.budget,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
