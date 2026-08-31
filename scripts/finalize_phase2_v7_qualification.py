"""CPU-only pair finalizer for two exact 4090D legacy-batch1 workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    CANARY_LEVELS,
    canonical_json_bytes,
    compare_canary_signatures,
    file_sha256,
    read_json,
    read_jsonl,
    validate_canary_audit,
    validate_legacy_backend_report,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v7_environment import (  # noqa: E402
    compare_environment_manifests,
)


def _eval_backend(contract: dict) -> dict:
    return dict(contract["eval_backend"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-gpu0", type=Path, required=True)
    parser.add_argument("--environment-gpu1", type=Path, required=True)
    parser.add_argument("--base-audit-gpu0", type=Path, required=True)
    parser.add_argument("--base-audit-gpu1", type=Path, required=True)
    parser.add_argument("--adapter-audit-gpu0", type=Path, required=True)
    parser.add_argument("--adapter-audit-gpu1", type=Path, required=True)
    parser.add_argument("--adapter-signatures-gpu0", type=Path, required=True)
    parser.add_argument("--adapter-signatures-gpu1", type=Path, required=True)
    parser.add_argument("--adapter-token-anchor", type=Path, required=True)
    parser.add_argument(
        "--backend-contract",
        type=Path,
        default=Path("configs/phase2_v7_legacy_batch1_contract.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    environments = [
        read_json(args.environment_gpu0.resolve()),
        read_json(args.environment_gpu1.resolve()),
    ]
    environment_contract_sha = compare_environment_manifests(
        baseline=environments[0], candidate=environments[1]
    )
    audits = {
        "gpu0": {
            "base_model_16": read_json(args.base_audit_gpu0.resolve()),
            "archived_adapter_16": read_json(args.adapter_audit_gpu0.resolve()),
        },
        "gpu1": {
            "base_model_16": read_json(args.base_audit_gpu1.resolve()),
            "archived_adapter_16": read_json(args.adapter_audit_gpu1.resolve()),
        },
    }
    for worker in ("gpu0", "gpu1"):
        for role in ("base_model_16", "archived_adapter_16"):
            validate_canary_audit(
                audit=audits[worker][role],
                expected_role=role,
                environment_contract_sha256=environment_contract_sha,
            )
        if audits[worker]["base_model_16"].get("comparison_levels") != list(
            CANARY_LEVELS
        ):
            raise ValueError(f"{worker} base canary did not compare every level")
    anchor = read_jsonl(args.adapter_token_anchor.resolve())
    for worker, path in (
        ("gpu0", args.adapter_signatures_gpu0),
        ("gpu1", args.adapter_signatures_gpu1),
    ):
        comparison = compare_canary_signatures(
            reference=anchor,
            candidate=read_jsonl(path.resolve()),
            comparison_levels=CANARY_LEVELS,
        )
        if comparison["status"] != "PASS":
            raise ValueError(f"{worker} adapter differs from the token anchor")
    backend = read_json(args.backend_contract.resolve())
    output_dir = args.output_dir.resolve()
    worker_reports = {}
    for worker in ("gpu0", "gpu1"):
        report = {
            "schema_version": "phase2-v7-legacy-backend-validation-v1",
            "status": "LEGACY_BATCH1_VALIDATED",
            "worker_id": worker,
            "gpu_uuid": environments[int(worker[-1])]["gpu"]["uuid"],
            "environment_contract_sha256": environment_contract_sha,
            "environment_manifest_sha256": file_sha256(
                (args.environment_gpu0 if worker == "gpu0" else args.environment_gpu1).resolve()
            ),
            "eval_backend": _eval_backend(backend),
            "batch_gt1_authorized": False,
            "canaries": audits[worker],
            "adapter_token_anchor_sha256": file_sha256(
                args.adapter_token_anchor.resolve()
            ),
            "accuracy_withheld": True,
        }
        path = output_dir / f"{worker}_legacy_backend_report.json"
        write_exclusive_or_verify(path, canonical_json_bytes(report))
        validate_legacy_backend_report(
            report_path=path, expected_sha256=file_sha256(path)
        )
        worker_reports[worker] = {
            "path": path.name,
            "sha256": file_sha256(path),
        }
    pair = {
        "schema_version": "phase2-v7-dual-worker-qualification-v1",
        "status": "PASS",
        "environment_contract_sha256": environment_contract_sha,
        "environment_manifests": {
            "gpu0": file_sha256(args.environment_gpu0.resolve()),
            "gpu1": file_sha256(args.environment_gpu1.resolve()),
        },
        "adapter_token_anchor_sha256": file_sha256(args.adapter_token_anchor.resolve()),
        "worker_reports": worker_reports,
        "batch_gt1_authorized": False,
        "formal_matrix_authorized": True,
        "accuracy_withheld": True,
        "gpu_accessed_by_finalizer": False,
    }
    write_exclusive_or_verify(
        output_dir / "dual_worker_qualification.json", canonical_json_bytes(pair)
    )
    print(json.dumps(pair, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
