from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.generator import SPLIT_SIZES, generate_all, generate_split  # noqa: E402
from eg_sft.utils.io import write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SPLIT_SIZES), default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    output_dir = ROOT / "data" / "samples"
    if args.all or args.split is None:
        generated = generate_all(seed=args.seed)
        for split, examples in generated.items():
            write_jsonl(output_dir / f"{split}.jsonl", (ex.to_dict() for ex in examples))
            print(f"wrote {split}: {len(examples)}")
    else:
        examples = generate_split(args.split, seed=args.seed)
        write_jsonl(output_dir / f"{args.split}.jsonl", (ex.to_dict() for ex in examples))
        print(f"wrote {args.split}: {len(examples)}")


if __name__ == "__main__":
    main()
