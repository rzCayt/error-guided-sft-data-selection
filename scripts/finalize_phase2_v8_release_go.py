"""Create a READY_FOR_HUMAN_REVIEW record after Q0-Q2 pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    write_exclusive_or_verify,
)
from eg_sft.evaluation.phase2_v8_canary import validate_v8_backend_report  # noqa: E402
from eg_sft.experiment.phase2_v8_canonical_runtime import (  # noqa: E402
    require_canonical_role,
    validate_canonical_runtime,
)
from eg_sft.experiment.phase2_v8_environment import (  # noqa: E402
    compare_v8_environment_manifests,
    validate_v8_environment_manifest,
)
from eg_sft.experiment.phase2_v8_release_gate import (  # noqa: E402
    require_clean_git,
    validate_deployment_tree,
)


def _require_sha(actual: str, expected: str, label: str) -> None:
    if len(expected) != 64 or actual != expected:
        raise ValueError(f"v8 release binding mismatch: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-runtime", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--release-archive", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--statistics-protocol", type=Path, required=True)
    parser.add_argument("--model-tree-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-tree-manifest", type=Path, required=True)
    parser.add_argument("--inference-final", type=Path, required=True)
    parser.add_argument("--training-anchor-final", type=Path, required=True)
    for worker in ("gpu0", "gpu1"):
        parser.add_argument(f"--environment-{worker}", type=Path, required=True)
        parser.add_argument(f"--backend-{worker}", type=Path, required=True)
    parser.add_argument("--q0-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical_path = args.canonical_runtime.resolve()
    canonical = validate_canonical_runtime(
        repo_root=ROOT, manifest_path=canonical_path
    )
    matrix_path = args.matrix.resolve()
    statistics_path = args.statistics_protocol.resolve()
    require_canonical_role(
        canonical=canonical, role="primary_matrix", actual_path=matrix_path
    )
    require_canonical_role(
        canonical=canonical, role="statistical_protocol", actual_path=statistics_path
    )

    deployment_path = args.deployment_manifest.resolve()
    release_archive_path = args.release_archive.resolve()
    deployment = validate_deployment_tree(
        repo_root=ROOT, manifest_path=deployment_path
    )
    git_commit = require_clean_git(ROOT)

    q0 = read_json(args.q0_gate.resolve())
    if q0.get("schema_version") != "phase2-v8-q0-cpu-gate-v1" or q0.get("status") != "PASS":
        raise ValueError("v8 Q0 gate has not passed")
    _require_sha(
        canonical["manifest_sha256"],
        str(q0.get("canonical_runtime_sha256", "")),
        "q0 canonical runtime",
    )
    _require_sha(
        deployment["manifest_sha256"],
        str(q0.get("deployment_manifest_sha256", "")),
        "q0 deployment manifest",
    )
    _require_sha(
        file_sha256(release_archive_path),
        str(q0.get("release_archive_sha256", "")),
        "q0 release archive",
    )
    if q0.get("git_commit") != git_commit:
        raise ValueError("v8 Q0 gate belongs to another Git commit")

    inference_path = args.inference_final.resolve()
    anchor_path = args.training_anchor_final.resolve()
    inference = read_json(inference_path)
    anchor = read_json(anchor_path)
    if (
        inference.get("status") != "PASS"
        or inference.get("formal_matrix_authorized") is not False
        or inference.get("canary_contract_sha256")
        != canonical["roles"]["canary_contract"]["sha256"]
    ):
        raise ValueError("v8 inference qualification is incomplete or unbound")
    if (
        anchor.get("status") != "PASS"
        or anchor.get("qualification_passed") is not True
        or anchor.get("training_anchor_protocol_sha256")
        != canonical["roles"]["training_anchor_protocol"]["sha256"]
        or anchor.get("canonical_runtime_sha256") != canonical["manifest_sha256"]
        or anchor.get("materialized_contracts_sha256")
        != canonical["roles"]["materialized_contracts"]["sha256"]
    ):
        raise ValueError("v8 training anchor qualification is incomplete or unbound")

    workers: dict[str, dict[str, str]] = {}
    environment_payloads = []
    for worker in ("gpu0", "gpu1"):
        environment_path = getattr(args, f"environment_{worker}").resolve()
        backend_path = getattr(args, f"backend_{worker}").resolve()
        environment = read_json(environment_path)
        environment_contract_sha = validate_v8_environment_manifest(environment)
        if environment.get("worker_id") != worker:
            raise ValueError("v8 environment worker identity changed")
        if environment.get("research", {}).get("phase2_matrix_sha256") != file_sha256(matrix_path):
            raise ValueError("v8 environment matrix binding changed")
        if (
            environment.get("research", {}).get("semantic_code_manifest_sha256")
            != canonical["roles"]["semantic_code_manifest"]["sha256"]
        ):
            raise ValueError("v8 environment semantic-code binding changed")
        backend = validate_v8_backend_report(
            report_path=backend_path,
            expected_sha256=file_sha256(backend_path),
            expected_worker_id=worker,
            expected_gpu_uuid=str(environment["gpu"]["uuid"]),
        )
        if backend.get("canary_contract_sha256") != canonical["roles"]["canary_contract"]["sha256"]:
            raise ValueError("v8 backend canary contract changed")
        expected_report = inference.get("worker_reports", {}).get(worker, {})
        if expected_report.get("sha256") != file_sha256(backend_path):
            raise ValueError("v8 inference final does not bind worker backend")
        workers[worker] = {
            "environment_manifest_sha256": file_sha256(environment_path),
            "environment_contract_sha256": environment_contract_sha,
            "backend_report_sha256": file_sha256(backend_path),
            "gpu_uuid": str(environment["gpu"]["uuid"]),
        }
        environment_payloads.append(environment)

    shared_environment_sha = compare_v8_environment_manifests(
        baseline=environment_payloads[0], candidate=environment_payloads[1]
    )
    if inference.get("environment_contract_sha256") != shared_environment_sha:
        raise ValueError("v8 inference/environment contract mismatch")
    if anchor.get("environment_contract_sha256") != shared_environment_sha:
        raise ValueError("v8 anchor/environment contract mismatch")

    payload = {
        "schema_version": "phase2-v8-release-go-v2",
        "status": "READY_FOR_HUMAN_REVIEW",
        "protocol_id": "phase2-clean-common24-v8",
        "git_commit": git_commit,
        "human_authorization": None,
        "bindings": {
            "canonical_runtime": canonical["manifest_sha256"],
            "deployment_manifest": deployment["manifest_sha256"],
            "release_archive": file_sha256(release_archive_path),
            "training_anchor_final": file_sha256(anchor_path),
            "model_tree_manifest": file_sha256(args.model_tree_manifest.resolve()),
            "tokenizer_tree_manifest": file_sha256(args.tokenizer_tree_manifest.resolve()),
            "matrix": file_sha256(matrix_path),
            "statistics_protocol": file_sha256(statistics_path),
            "canary_contract": canonical["roles"]["canary_contract"]["sha256"],
            "training_anchor_protocol": canonical["roles"]["training_anchor_protocol"]["sha256"],
            "semantic_code_manifest": canonical["roles"]["semantic_code_manifest"]["sha256"],
            "materialized_contracts": canonical["roles"]["materialized_contracts"]["sha256"],
            "inference_final": file_sha256(inference_path),
            "q0_gate": file_sha256(args.q0_gate.resolve()),
        },
        "workers": workers,
        "deployment_file_count": deployment["file_count"],
        "accuracy_blind": True,
        "formal_matrix_authorized": False,
        "required_human_authorization": "START_PHASE2_V8_COMMON24",
    }
    write_exclusive_or_verify(args.output.resolve(), canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
