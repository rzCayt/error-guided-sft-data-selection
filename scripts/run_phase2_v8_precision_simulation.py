"""Freeze the prospective v8 precision sensitivity result before GPU use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v8_statistics import (  # noqa: E402
    precision_sensitivity_grid,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    report = precision_sensitivity_grid(
        simulations=args.simulations,
        seed=args.seed,
    )
    output = args.output.resolve()
    write_bytes_exclusive_or_verify(output, canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "equivalence_status": report["equivalence_status"],
                "scenario_count": len(report["rows"]),
                "sha256": file_sha256(output),
                "gpu_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
