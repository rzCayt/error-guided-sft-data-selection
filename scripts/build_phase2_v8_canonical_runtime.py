"""Generate the sole allowed v8 runtime file authority after code freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


FIXED_ROLES = {
    "primary_matrix": "configs/phase2_clean_common24_v8_canonical.json",
    "statistical_protocol": "configs/phase2_v8_statistical_protocol.json",
    "training_anchor_protocol": "configs/phase2_v8_training_anchor_protocol.json",
    "canary_contract": "configs/phase2_v8_canary_contract.json",
    "stop_go_rules": "configs/phase2_v8_stop_go_rules.json",
    "parent_matrix": "configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json",
    "base_recipe": "configs/budget_equivalent_lora_v3.json",
    "research_protocol": "configs/public_gsm8k_v1.json",
    "information_gates": ".aris/compute/budget_equivalent_v3_selections/information_gates.json",
    "materialized_contracts": "artifacts/phase2_v8_materialized_contracts_v4/MATERIALIZATION_COMPLETE.json",
    "materialized_contract_audit": "artifacts/phase2_v8_preflight/materialized_contract_audit_v4.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-code-manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json"),
    )
    args = parser.parse_args()
    roles = dict(FIXED_ROLES)
    semantic = args.semantic_code_manifest.resolve()
    semantic.relative_to(ROOT)
    expected_semantic = (
        ROOT / "artifacts/phase2_v8_preflight/semantic_code_manifest_v8_2.json"
    ).resolve()
    if semantic != expected_semantic:
        raise ValueError("v8 canonical builder requires the frozen v8.2 semantic manifest")
    output = args.output.resolve()
    expected_output = (ROOT / "configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json").resolve()
    if output != expected_output:
        raise ValueError("v8 canonical builder cannot create an alternate authority")
    roles["semantic_code_manifest"] = semantic.relative_to(ROOT).as_posix()
    files = []
    for role, relative in sorted(roles.items()):
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        if not path.is_file():
            raise ValueError(f"canonical runtime file is missing: {role}")
        files.append({"role": role, "path": relative, "sha256": file_sha256(path)})
    payload = {
        "schema_version": "phase2-v8-canonical-runtime-files-v1",
        "protocol_id": "phase2-clean-common24-v8",
        "status": "FROZEN",
        "noncanonical_config_files_are_non_deployable": True,
        "files": files,
    }
    write_bytes_exclusive_or_verify(output, canonical_json_bytes(payload))
    print(json.dumps({"status": "PASS", "file_count": len(files), "sha256": file_sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
