"""Compatibility wrapper that gives the validated GSM8K worker a v3 contract."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import _read_json  # noqa: E402

import eg_sft.experiment.cloud_v2_formal as legacy_contract  # noqa: E402
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    resolve_phase1_contract,
)


def _argument(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing required wrapper argument: {name}") from error


def main() -> None:
    run_dir = Path(_argument("--run-dir")).resolve()
    _argument("--config")
    manifest = _read_json(run_dir / "manifest.json")
    cell_id = str(manifest["config"]["cell_id"])

    def _resolve(*, repo_root: Path, config_path: Path, method: str, seed: int):
        contract = resolve_phase1_contract(
            repo_root=repo_root,
            config_path=config_path,
            cell_id=cell_id,
        )
        if contract["method"] != method or contract["seed"] != seed:
            raise ValueError("worker method/seed differs from frozen v3 cell")
        return contract

    legacy_contract.resolve_formal_contract = _resolve
    import run_cloud_v2_formal_eval_worker

    run_cloud_v2_formal_eval_worker.resolve_formal_contract = _resolve
    run_cloud_v2_formal_eval_worker.main()


if __name__ == "__main__":
    main()
