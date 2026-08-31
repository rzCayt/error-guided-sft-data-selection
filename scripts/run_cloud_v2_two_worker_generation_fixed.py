"""Fixed-config launcher for two batch-one processes on one CUDA device."""

from __future__ import annotations

import sys

from run_cloud_v2_two_worker_generation import main


def _inject_fixed_config() -> None:
    if "--config" not in sys.argv:
        sys.argv[1:1] = [
            "--config",
            "configs/cloud_v2_two_worker_generation_fixed_v1.json",
        ]


if __name__ == "__main__":
    _inject_fixed_config()
    main()
