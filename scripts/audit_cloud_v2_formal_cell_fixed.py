"""Fixed-contract entry for one accuracy-blind formal cell audit."""

from __future__ import annotations

import sys

from audit_cloud_v2_formal_cell import main


def _inject_fixed_config() -> None:
    if "--config" not in sys.argv:
        sys.argv[1:1] = [
            "--config",
            "configs/cloud_v2_formal_b500_single_cell_fixed_v1.json",
        ]


if __name__ == "__main__":
    _inject_fixed_config()
    main()
