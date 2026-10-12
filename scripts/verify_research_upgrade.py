"""Validate new-study statuses, budget and immutable historical bindings on CPU."""
from __future__ import annotations

import json

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()
from eg_sft.experiment.research_upgrade import (  # noqa: E402
    read_object, validate_budget, validate_registry,
)


def main() -> None:
    registry = read_object(ROOT / "results/research_upgrade/registry.json")
    validate_registry(registry, ROOT)
    used = validate_budget(read_object(ROOT / "configs/research_upgrade_budget.json"))
    historical = read_object(ROOT / "results/public_summary/main_results.json")
    if historical["state_dependence_v3"]["gpu_result_available"] is not False:
        raise ValueError("Do not rewrite historical unavailable measurements as new results")
    print(json.dumps({"status": "PASS", "research_pilots": 3,
                      "study_count": len(registry["studies"]),
                      "committed_budget_cny": str(sum(used.values())),
                      "gpu_results": sum(row["gpu_result_available"] for row in registry["studies"])}))


if __name__ == "__main__":
    main()
