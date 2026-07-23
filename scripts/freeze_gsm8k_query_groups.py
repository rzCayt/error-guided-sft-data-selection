"""Freeze text-free all-query and error-query lists for selector research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eg_sft.selection.query_groups import (
    freeze_query_groups,
    load_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k-records", type=Path, required=True)
    parser.add_argument("--diagnostic-outputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-group-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {args.output_dir}"
        )

    split_records = load_jsonl(args.gsm8k_records)
    diagnostic_outputs = load_jsonl(args.diagnostic_outputs)
    all_queries, error_queries, manifest = freeze_query_groups(
        split_records=split_records,
        diagnostic_outputs=diagnostic_outputs,
        minimum_group_size=args.minimum_group_size,
    )

    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "all_queries.jsonl", all_queries)
    write_jsonl(args.output_dir / "error_queries.jsonl", error_queries)
    write_json(args.output_dir / "query_group_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
