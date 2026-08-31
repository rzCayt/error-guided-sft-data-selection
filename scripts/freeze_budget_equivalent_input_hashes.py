"""Create a new immutable protocol config bound to prepared v3 input hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_protocol import read_json_object  # noqa: E402
from eg_sft.training.b500 import file_sha256  # noqa: E402


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/budget_equivalent_v3_protocol.json"),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    payload = read_json_object(args.base_config.resolve())
    input_dir = args.input_dir.resolve()
    similarity = input_dir / "query_candidate_similarity.pt"
    clusters = input_dir / "near_duplicate_clusters.jsonl"
    if not similarity.is_file() or not clusters.is_file():
        raise FileNotFoundError("both similarity and near-duplicate clusters are required")
    payload["similarity_artifact"] = {
        **payload["similarity_artifact"],
        "path": _relative(similarity),
        "sha256": file_sha256(similarity),
    }
    payload["near_duplicate_clusters"] = {
        **payload["near_duplicate_clusters"],
        "path": _relative(clusters),
        "sha256": file_sha256(clusters),
    }
    with args.output_config.resolve().open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(file_sha256(args.output_config.resolve()))


if __name__ == "__main__":
    main()
