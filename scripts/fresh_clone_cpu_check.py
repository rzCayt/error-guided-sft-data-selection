#!/usr/bin/env python3
"""Clone the committed repository into a temporary directory and run CPU gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


COMMANDS = (
    ("scripts/reproduce_public_summary.py", "--check"),
    ("scripts/snapshot_public_configs.py", "--check"),
    ("scripts/generate_experiment_registry.py", "--check"),
    ("figures/generate_public_figures.py", "--check"),
    ("scripts/mark_historical_snapshots.py", "--check"),
    ("scripts/verify_public_release.py",),
)


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.joinpath(".git").exists():
        raise ValueError(f"source is not a Git checkout: {source}")

    with tempfile.TemporaryDirectory(prefix="eg-sft-public-clone-") as temp_text:
        clone = Path(temp_text) / "repo"
        run(["git", "clone", "--no-hardlinks", str(source), str(clone)], cwd=source.parent)
        venv = Path(temp_text) / "venv"
        run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], cwd=clone)
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        run([str(python), "-m", "pip", "install", "--no-deps", "-e", "."], cwd=clone)
        for command in COMMANDS:
            run([str(python), *command], cwd=clone)
        run([str(python), "-m", "pytest", "-q"], cwd=clone)
        run([str(python), "-m", "ruff", "check", "."], cwd=clone)
    print("PASS fresh_clone_cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
