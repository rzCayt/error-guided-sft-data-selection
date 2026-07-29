"""Print the frozen nine-job B=500 plan without launching any GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.b500_matrix import preflight_b500_matrix  # noqa: E402


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("matrix config must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=Path("configs/b500_formal_matrix_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally save the dry-run report; an existing file is never replaced.",
    )
    args = parser.parse_args()

    spec = _read_json(args.matrix_config.resolve())
    report = preflight_b500_matrix(spec=spec, repo_root=ROOT)
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
