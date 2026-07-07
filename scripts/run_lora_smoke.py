from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.training.lora_smoke import run_lora_smoke  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="results/lora_smoke")
    args = parser.parse_args()

    evidence = run_lora_smoke(args.model, ROOT / args.output_dir)
    print(f"wrote {evidence}")


if __name__ == "__main__":
    main()
