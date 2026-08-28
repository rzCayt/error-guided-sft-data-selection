"""CPU-only final gate for the two v4 worker registries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.identifiable_budget_v4 import identifiable_registry  # noqa: E402
from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/identifiable_budget_v4_matrix.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    registry = identifiable_registry(repo_root=ROOT, config_path=config_path)
    ready = registry["audited_pass_count"] == registry["job_count"] == 12
    artifact = {
        **registry,
        "status": "AUDITED_PASS" if ready else "BLOCKED_INCOMPLETE",
        "unblinding_permitted": ready,
        "automatic_unblinding": False,
        "claim_boundary": "Integrity gate only; this artifact contains no accuracy comparison.",
    }
    output = args.output.resolve()
    write_bytes_exclusive_or_verify(output, canonical_json_bytes(artifact))
    write_bytes_exclusive_or_verify(
        output.with_suffix(output.suffix + ".sha256"),
        f"{file_sha256(output)}  {output.name}\n".encode("ascii"),
    )
    print(json.dumps({"status": artifact["status"], "audited_pass_count": registry["audited_pass_count"], "job_count": 12, "accuracy_withheld": True}, sort_keys=True))


if __name__ == "__main__":
    main()
