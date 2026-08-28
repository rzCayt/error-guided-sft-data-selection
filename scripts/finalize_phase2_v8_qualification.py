"""CPU-only finalizer for v8 environment and three-layer inference bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    read_jsonl,
    write_exclusive_or_verify,
)
from eg_sft.evaluation.phase2_v8_canary import (  # noqa: E402
    FULL_LEVELS,
    compare_v8_signatures,
    validate_v8_backend_report,
)
from eg_sft.experiment.phase2_v8_environment import (  # noqa: E402
    compare_v8_environment_manifests,
)


def _passed_audit(path: Path, *, role: str, count: int, environment_sha: str) -> dict:
    audit = read_json(path.resolve())
    if (
        audit.get("schema_version") != "phase2-v8-canary-audit-v1"
        or audit.get("status") != "PASS"
        or audit.get("role") != role
        or int(audit.get("record_count", -1)) != count
        or audit.get("environment_contract_sha256") != environment_sha
        or audit.get("historical_bridge", {}).get("status") != "PASS"
        or audit.get("historical_token_exact_claimed") is not False
    ):
        raise ValueError(f"v8 canary audit changed: {role}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    for worker in ("gpu0", "gpu1"):
        parser.add_argument(f"--environment-{worker}", type=Path, required=True)
        parser.add_argument(f"--base-audit-{worker}", type=Path, required=True)
        parser.add_argument(f"--base-signatures-{worker}", type=Path, required=True)
        parser.add_argument(f"--adapter-audit-{worker}", type=Path, required=True)
        parser.add_argument(f"--adapter-signatures-{worker}", type=Path, required=True)
    parser.add_argument(
        "--canary-contract", type=Path, default=Path("configs/phase2_v8_canary_contract.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    env_paths = [args.environment_gpu0.resolve(), args.environment_gpu1.resolve()]
    environments = [read_json(path) for path in env_paths]
    environment_sha = compare_v8_environment_manifests(
        baseline=environments[0], candidate=environments[1]
    )
    audits = {}
    for worker in ("gpu0", "gpu1"):
        audits[worker] = {
            "base16": _passed_audit(
                getattr(args, f"base_audit_{worker}"),
                role="base16",
                count=16,
                environment_sha=environment_sha,
            ),
            "adapter128": _passed_audit(
                getattr(args, f"adapter_audit_{worker}"),
                role="adapter128",
                count=128,
                environment_sha=environment_sha,
            ),
        }
    base_exact = compare_v8_signatures(
        reference=read_jsonl(args.base_signatures_gpu0.resolve()),
        candidate=read_jsonl(args.base_signatures_gpu1.resolve()),
        levels=FULL_LEVELS,
        expected_count=16,
    )
    adapter_exact = compare_v8_signatures(
        reference=read_jsonl(args.adapter_signatures_gpu0.resolve()),
        candidate=read_jsonl(args.adapter_signatures_gpu1.resolve()),
        levels=FULL_LEVELS,
        expected_count=128,
    )
    if base_exact["status"] != "PASS" or adapter_exact["status"] != "PASS":
        raise ValueError("v8 new-block token-exact comparison failed")
    canary_contract_path = args.canary_contract.resolve()
    canary_contract = read_json(canary_contract_path)
    canary_contract_sha256 = file_sha256(canary_contract_path)
    backend = canary_contract["backend"]
    output_dir = args.output_dir.resolve()
    reports = {}
    for index, worker in enumerate(("gpu0", "gpu1")):
        report = {
            "schema_version": "phase2-v8-legacy-backend-validation-v1",
            "status": "LEGACY_BATCH1_VALIDATED",
            "worker_id": worker,
            "gpu_uuid": environments[index]["gpu"]["uuid"],
            "environment_contract_sha256": environment_sha,
            "environment_manifest_sha256": file_sha256(env_paths[index]),
            "eval_backend": backend,
            "canary_contract_sha256": canary_contract_sha256,
            "batch_gt1_authorized": False,
            "base_new_block_exact": True,
            "adapter_historical_semantic_bridge": True,
            "adapter_new_block_token_exact": True,
            "historical_token_exact_claimed": False,
            "base_cross_worker_comparison": base_exact,
            "adapter_cross_worker_comparison": adapter_exact,
            "canary_audit_sha256": {
                "base16": file_sha256(getattr(args, f"base_audit_{worker}").resolve()),
                "adapter128": file_sha256(getattr(args, f"adapter_audit_{worker}").resolve()),
            },
            "accuracy_withheld": True,
        }
        path = output_dir / f"{worker}_v8_legacy_backend_report.json"
        write_exclusive_or_verify(path, canonical_json_bytes(report))
        validate_v8_backend_report(report_path=path, expected_sha256=file_sha256(path))
        reports[worker] = {"path": path.name, "sha256": file_sha256(path)}
    pair = {
        "schema_version": "phase2-v8-dual-worker-inference-qualification-v1",
        "status": "PASS",
        "environment_contract_sha256": environment_sha,
        "canary_contract_sha256": canary_contract_sha256,
        "base16_cross_worker_exact": True,
        "adapter128_historical_semantic_bridge": True,
        "adapter128_cross_worker_token_exact": True,
        "historical_token_exact_claimed": False,
        "worker_reports": reports,
        "training_anchor_still_required": True,
        "formal_matrix_authorized": False,
        "gpu_accessed_by_finalizer": False,
        "accuracy_withheld": True,
    }
    write_exclusive_or_verify(
        output_dir / "v8_inference_qualification.json", canonical_json_bytes(pair)
    )
    print(json.dumps(pair, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
