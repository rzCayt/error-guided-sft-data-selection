"""Build immutable, ASCII-only v8 deployment and independent-review archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


WINDOWS_PRIVATE_PATH = re.compile(rb"[A-Za-z]:\\(?:Users|RA)[^\r\n\"']*")
SSH_ENDPOINT = re.compile(rb"(?:ssh\s+-p\s+\d+|root@connect\.[^\s]+)", re.I)
TOKEN_VALUE = re.compile(
    rb"(?:password|api[_-]?key|access[_-]?token|secret)[ \t]*[:=][ \t]*[\"'][A-Za-z0-9+/_.-]{12,}[\"']|(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{12,}",
    re.I,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_ascii(relative: str) -> None:
    relative.encode("ascii")
    if relative.startswith("/") or ".." in Path(relative).parts or "\\" in relative:
        raise ValueError(f"unsafe package path: {relative}")


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file() and "__pycache__" not in child.parts and child.suffix not in {
            ".pyc",
            ".pyo",
        }:
            yield child


def _scan(relative: str, payload: bytes) -> None:
    _safe_ascii(relative)
    if Path(relative).suffix.lower() in {".safetensors", ".pt", ".bin"}:
        return
    # This literal is an intentional unit-test fixture for path redaction, not
    # a path from the user's machine. Keep the test source intact in the review
    # package while excluding the synthetic placeholder from privacy findings.
    scan_payload = payload.replace(rb"C:\Users\person", b"<TEST_WINDOWS_PATH>")
    for label, pattern in (
        ("private_windows_path", WINDOWS_PRIVATE_PATH),
        ("ssh_endpoint", SSH_ENDPOINT),
        ("secret_like_value", TOKEN_VALUE),
    ):
        if pattern.search(scan_payload):
            raise ValueError(f"{label} found in package file: {relative}")


def _manifest(files: list[tuple[str, bytes]], *, kind: str) -> dict:
    rows = [
        {"path": relative, "size_bytes": len(payload), "sha256": _sha256(payload)}
        for relative, payload in files
    ]
    return {
        "schema_version": "phase2-v8-release-manifest-v1",
        "protocol_id": "phase2-clean-common24-v8",
        "kind": kind,
        "file_count": len(rows),
        "files": rows,
        "ascii_paths_only": True,
        "private_windows_path_hits": 0,
        "ssh_endpoint_hits": 0,
        "secret_like_value_hits": 0,
        "gpu_accessed": False,
    }


def _fixed_configs() -> list[Path]:
    names = (
        "phase2_clean_common24_v8_canonical.json",
        "CANONICAL_RUNTIME_FILES_v8_RELEASE.json",
        "phase2_v8_statistical_protocol.json",
        "phase2_v8_training_anchor_protocol.json",
        "phase2_v8_canary_contract.json",
        "phase2_v8_stop_go_rules.json",
        "budget_equivalent_phase1_matrix_frozen_20260824_v2.json",
        "budget_equivalent_phase1_matrix_frozen_20260824.json",
        "budget_equivalent_lora_v3.json",
        "budget_equivalent_ood_v1.json",
        "public_gsm8k_v1.json",
    )
    return [ROOT / "configs" / name for name in names]


def _deployment_sources(cpu_validation: Path) -> list[Path]:
    roots = (
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "workflow",
        ROOT / "data",
        ROOT / "results/public_release_v1",
        ROOT / "results/residual_selector_identifiability",
        ROOT / "results/model_aware_signal_f2",
        ROOT / "results/strong_baseline_protocol_v2_ab_rescore",
        ROOT / "results/research_public_gsm8k_v1/data_manifest_full_v2_fuzzy",
        ROOT / "results/research_public_gsm8k_v1/h1a_formal_tulu96_clean_3fdb8b5",
        ROOT / "results/research_public_gsm8k_v1/b500_random_selection_v1",
        ROOT / "results/research_public_gsm8k_v1/b500_rds_all_selection_v1",
        ROOT / "results/research_public_gsm8k_v1/b500_rds_error_selection_v1",
        ROOT / "results/research_public_gsm8k_v1/rds_full_pool_10k_public_evidence_v1",
        ROOT / ".aris/compute",
        ROOT / "artifacts/phase2_v7_canary",
        ROOT / "artifacts/phase2_v8_canary",
        ROOT / "artifacts/phase2_v8_materialized_contracts_v4",
        ROOT / "artifacts/phase2_v8_parent_evidence",
        ROOT / "tests/fixtures/phase2_v7_anchor/training_complete/adapter",
        ROOT / "tests/fixtures/phase2_v7_anchor/training_complete/tokenizer",
    )
    exact = [
        ROOT / "pyproject.toml",
        ROOT / "requirements-budget-equivalent.txt",
        ROOT / "requirements-cloud-b500.txt",
        ROOT / "artifacts/phase2_v8_preflight/precision_simulation.json",
        ROOT / "artifacts/phase2_v8_preflight/semantic_code_manifest_v8_2.json",
        ROOT / "artifacts/phase2_v8_preflight/materialized_contract_audit_v4.json",
        ROOT / "artifacts/phase2_v8_preflight/experiment_audit_p0_v3.json",
        ROOT / "artifacts/phase2_v8_preflight/final_gate_v5/preflight_report.json",
        ROOT / "artifacts/phase2_v8_preflight/final_gate_v5/commands.jsonl",
        ROOT / "docs/history/legacy_repository_docs/20260828_PHASE2_V8_LONG_EXPERIMENT_READINESS.md",
        ROOT / "docs/history/legacy_repository_docs/20260828_PHASE2_V8_EXPERIMENT_AUDIT.md",
        ROOT / "docs/history/legacy_repository_docs/20260828_PHASE2_V8_GPTPRO_REVIEW_PROMPT.md",
        ROOT / "releases/phase2_v8/20260828_PHASE2_V8_RELEASE_README.md",
        ROOT / "docs/history/legacy_repository_docs/20260828_PHASE2_V8_2_FINAL_AUDIT.md",
        ROOT / "docs/history/legacy_repository_docs/20260828_PHASE2_V8_2_START_HERE.md",
        ROOT / "docs/history/legacy_repository_docs/20260828_PHASE2_V8_2_CODEX_EXECUTION_DIRECTIVE.md",
        ROOT / "docs/history/legacy_repository_docs/workflow_cn.md",
        ROOT / "docs/history/legacy_repository_docs/adversarial_review_rubric_cn.md",
        ROOT / "docs/history/legacy_repository_docs/reviewer_external_evidence_policy_cn.md",
        ROOT / "docs/history/legacy_repository_docs/workflow.md",
        ROOT / "docs/history/legacy_repository_docs/adversarial_review_protocol.md",
        cpu_validation.resolve(),
        ROOT / "artifacts/phase2_v8_preflight/v8_2_local_validation/targeted_v8_pytest.log",
        ROOT / "artifacts/phase2_v8_preflight/v8_2_local_validation/full_pytest.log",
        *_fixed_configs(),
    ]
    files = list(exact)
    for root in roots:
        files.extend(_iter_files(root))
    return sorted(set(path.resolve() for path in files))


def _collect(sources: Iterable[Path]) -> list[tuple[str, bytes]]:
    files = []
    for source in sources:
        if not source.is_file():
            raise ValueError(f"v8 release source missing: {source}")
        relative = source.relative_to(ROOT).as_posix()
        payload = source.read_bytes()
        _scan(relative, payload)
        files.append((relative, payload))
    return files


def _write_tar(path: Path, files: list[tuple[str, bytes]]) -> dict:
    manifest = _manifest(files, kind="autodl_deployment")
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    sums = "".join(
        f"{_sha256(payload)}  {relative}\n" for relative, payload in files
    ).encode("ascii")
    release_readme = (
        ROOT / "releases/phase2_v8/20260828_PHASE2_V8_RELEASE_README.md"
    ).read_bytes()
    with tarfile.open(path, "x:gz", compresslevel=6) as archive:
        for relative, payload in files:
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
        info = tarfile.TarInfo("DEPLOYMENT_MANIFEST.json")
        info.size = len(manifest_raw)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(manifest_raw))
        for name, payload in (
            ("SHA256SUMS.txt", sums),
            ("RELEASE_README.md", release_readme),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return manifest


def _write_review(path: Path, deployment_manifest: dict, files: list[tuple[str, bytes]]) -> dict:
    selected = list(files)
    deployment_raw = (
        json.dumps(deployment_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    selected.append(
        (
            "review/deployment_manifest.json",
            deployment_raw,
        )
    )
    selected.append(("DEPLOYMENT_MANIFEST.json", deployment_raw))
    selected.append(
        (
            "SHA256SUMS.txt",
            "".join(
                f"{_sha256(payload)}  {relative}\n"
                for relative, payload in files
            ).encode("ascii"),
        )
    )
    selected.append(
        (
            "RELEASE_README.md",
            (ROOT / "releases/phase2_v8/20260828_PHASE2_V8_RELEASE_README.md").read_bytes(),
        )
    )
    manifest = _manifest(selected, kind="independent_review_with_actual_source")
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, payload in selected:
            archive.writestr(relative, payload)
        archive.writestr(
            "REVIEW_MANIFEST.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-validation", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    files = _collect(_deployment_sources(args.cpu_validation))
    deployment = output_dir / "phase2_v8_autodl_deployment.tar.gz"
    review = output_dir / "phase2_v8_gptpro_review.zip"
    deployment_manifest = _write_tar(deployment, files)
    review_manifest = _write_review(review, deployment_manifest, files)
    summary = {
        "schema_version": "phase2-v8-release-summary-v1",
        "status": "PASS",
        "protocol_id": "phase2-clean-common24-v8",
        "deployment": {
            "path": deployment.name,
            "sha256": _sha256(deployment.read_bytes()),
            "size_bytes": deployment.stat().st_size,
            "file_count": deployment_manifest["file_count"],
        },
        "review": {
            "path": review.name,
            "sha256": _sha256(review.read_bytes()),
            "size_bytes": review.stat().st_size,
            "file_count": review_manifest["file_count"],
        },
        "gpu_accessed": False,
    }
    (output_dir / "release_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
