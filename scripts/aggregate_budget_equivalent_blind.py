"""Report the blinded aggregation gate; never unblind or print accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_blind_aggregate import (  # noqa: E402
    guarded_blind_aggregation,
)
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    phase1_registry,
    read_json_object,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json_object(config_path)
    registry = phase1_registry(repo_root=ROOT, config_path=config_path)
    ood_status = {}
    for job in registry["jobs"]:
        status = "PENDING"
        if len(job["run_dirs"]) == 1:
            audit_path = Path(job["run_dirs"][0]) / "audit" / "ood_audit.json"
            if audit_path.is_file():
                audit = read_json_object(audit_path)
                status = "AUDITED_PASS" if audit.get("status") == "PASS" else "AUDIT_FAILED"
        ood_status[str(job["cell_id"])] = status
    payload = guarded_blind_aggregation(
        private_map=read_json_object(args.private_map.resolve()),
        registry=registry,
        ood_status_by_cell=ood_status,
        ood_required=bool(
            config.get("execution_policy", {}).get(
                "ood_audits_required_before_unblinding", False
            )
        ),
    )
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
