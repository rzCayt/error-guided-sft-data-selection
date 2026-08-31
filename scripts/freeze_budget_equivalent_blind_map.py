"""Freeze private/public blind mappings before any Phase 1 result exists."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_blind import build_blind_map  # noqa: E402
from eg_sft.experiment.budget_equivalent_matrix import read_json_object  # noqa: E402
from eg_sft.training.b500 import file_sha256  # noqa: E402


def _write_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--secret-hex")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    private, public = build_blind_map(
        matrix_config=read_json_object(config_path),
        matrix_sha256=file_sha256(config_path),
        secret_hex=args.secret_hex or secrets.token_hex(32),
    )
    private_path = output_dir / "private_blind_map.json"
    public_path = output_dir / "public_blind_manifest.json"
    _write_exclusive(private_path, private)
    public["private_map_sha256"] = file_sha256(private_path)
    _write_exclusive(public_path, public)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "matrix_sha256": public["matrix_sha256"],
                "private_map_sha256": public["private_map_sha256"],
                "public_manifest_sha256": file_sha256(public_path),
                "blind_cell_count": len(public["cells"]),
                "actual_method_names_withheld": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
