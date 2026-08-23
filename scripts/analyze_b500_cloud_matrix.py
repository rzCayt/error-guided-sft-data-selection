"""Analyze exactly nine audited cloud B=500 runs; never launch a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.b500_comparison import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    analyze_complete_matrix,
    load_complete_matrix,
)


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    with path.with_name(path.name + ".sha256").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(f"{digest}  {path.name}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "results/research_public_gsm8k_v1/b500_formal_cloud_4090_runs_v1"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output = args.output.resolve()
    matrix = load_complete_matrix(run_root)
    report = analyze_complete_matrix(
        matrix,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    try:
        run_root_label = run_root.relative_to(ROOT).as_posix()
    except ValueError:
        run_root_label = run_root.name
    report["provenance"] = {
        "analyzed_at_utc": datetime.now(UTC).isoformat(),
        "analysis_git_commit": _git_commit(),
        "command": [sys.executable, *sys.argv],
        "run_root": run_root_label,
    }
    _write_exclusive(output, report)
    print(json.dumps(report["frozen_downstream_gate"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
