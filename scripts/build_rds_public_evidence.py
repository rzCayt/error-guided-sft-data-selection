"""Build a path-clean public evidence package without rewriting local evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.rds_full_pool import (  # noqa: E402
    canonical_json_sha256,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


PUBLIC_SCHEMA = "rds-full-pool-public-evidence-v1"
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:^|[\"'\s])(?:[a-z]:[\\/])")
UNIX_PRIVATE_PATH = re.compile(r"(?:^|[\"'\s])/(?:home|Users)/[^/\"'\s]+/")
SECRET_LIKE = re.compile(
    r"(?i)(?:"
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    r"hf_[A-Za-z0-9]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"(?:api[_-]?key|access[_-]?token|secret)[\"'\s:=]+[A-Za-z0-9_./+-]{16,}"
    r")"
)
RAW_TEXT_KEY = re.compile(
    r'"(?:messages|question|answer|completion|generated_text|prompt_text)"\s*:'
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        for block in iter(lambda: input_handle.read(1024 * 1024), b""):
            output_handle.write(block)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside repository: {path}") from error


def _verify_json_self_hash(
    payload: dict[str, Any],
    *,
    field: str,
) -> None:
    claimed = payload.get(field)
    material = dict(payload)
    material.pop(field, None)
    if claimed != canonical_json_sha256(material):
        raise ValueError(f"{field} does not match payload")


def _scan_text(
    *,
    label: str,
    text: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    checks = (
        ("windows_absolute_path", WINDOWS_ABSOLUTE_PATH),
        ("unix_private_path", UNIX_PRIVATE_PATH),
        ("secret_like_value", SECRET_LIKE),
        ("raw_source_text_field", RAW_TEXT_KEY),
    )
    for finding_type, pattern in checks:
        if pattern.search(text):
            findings.append(
                {
                    "file": label,
                    "finding_type": finding_type,
                }
            )
    return findings


def _safe_contract(
    *,
    local_contract: dict[str, Any],
    local_contract_file_sha256: str,
) -> dict[str, Any]:
    command = list(local_contract["command"])
    if not command:
        raise ValueError("local command is empty")
    command[0] = "python"
    payload = {
        "schema_version": "rds-public-run-contract-v1",
        "scope": local_contract["scope"],
        "representation": local_contract["representation"],
        "thermal": local_contract["thermal"],
        "selection": local_contract["selection"],
        "protocol": local_contract["protocol"],
        "prepared_candidate_scope_count": local_contract[
            "prepared_candidate_scope_count"
        ],
        "prepared_query_scope_count": local_contract[
            "prepared_query_scope_count"
        ],
        "source_git_commit": local_contract["source_git_commit"],
        "input_bindings": local_contract["input_bindings"],
        "implementation_bindings": local_contract["implementation_bindings"],
        "command": command,
        "source_local_evidence": {
            "run_contract_self_sha256": local_contract[
                "run_contract_sha256"
            ],
            "run_contract_file_sha256": local_contract_file_sha256,
        },
        "sanitization": {
            "immutable_local_evidence_changed": False,
            "transformed_fields": ["command[0]"],
            "rule": (
                "Replace the machine-specific Python executable with the "
                "portable command name 'python'."
            ),
        },
        "claim_boundary": local_contract["claim_boundary"],
    }
    payload["public_contract_sha256"] = canonical_json_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rds-all-manifest", type=Path, required=True)
    parser.add_argument("--rds-error-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"public evidence directory exists: {output_dir}")
    contract_path = run_dir / "run_contract.json"
    prepared_path = run_dir / "prepared.json"
    finalization_path = run_dir / "finalization_manifest.json"
    local_contract = _read_json(contract_path)
    _verify_json_self_hash(local_contract, field="run_contract_sha256")
    prepared = _read_json(prepared_path)
    finalization = _read_json(finalization_path)
    if prepared.get("run_contract_sha256") != local_contract["run_contract_sha256"]:
        raise ValueError("prepared artifact run contract changed")
    if (
        finalization.get("run_contract_sha256")
        != local_contract["run_contract_sha256"]
    ):
        raise ValueError("finalization run contract changed")
    if local_contract.get("scope") != "formal_10000_candidate_pool":
        raise ValueError("public package requires the formal full-pool run")

    scores_source = run_dir / finalization["artifacts"]["candidate_scores"]["path"]
    metrics_source = run_dir / finalization["artifacts"]["metrics"]["path"]
    if file_sha256(scores_source) != finalization["artifacts"][
        "candidate_scores"
    ]["sha256"]:
        raise ValueError("candidate score hash changed")
    if file_sha256(metrics_source) != finalization["artifacts"]["metrics"]["sha256"]:
        raise ValueError("metrics hash changed")
    metrics = _read_json(metrics_source)

    selection_sources = {
        "rds_all": args.rds_all_manifest.resolve(),
        "rds_error": args.rds_error_manifest.resolve(),
    }
    selections: dict[str, dict[str, Any]] = {}
    for strategy, path in selection_sources.items():
        manifest = _read_json(path)
        _verify_json_self_hash(manifest, field="manifest_content_sha256")
        if manifest.get("strategy") != strategy:
            raise ValueError(f"{strategy} selection strategy changed")
        if len(manifest.get("selected_candidates", [])) != 500:
            raise ValueError(f"{strategy} selection count changed")
        selections[strategy] = manifest

    public_contract = _safe_contract(
        local_contract=local_contract,
        local_contract_file_sha256=file_sha256(contract_path),
    )
    data_provenance = {
        "schema_version": "rds-public-data-provenance-v1",
        "protocol": local_contract["protocol"],
        "input_bindings": local_contract["input_bindings"],
        "audited_candidate_count": int(prepared["audited_candidate_count"]),
        "eligible_candidate_count": int(prepared["eligible_candidate_count"]),
        "excluded_fully_truncated_count": int(
            prepared["excluded_fully_truncated_count"]
        ),
        "all_query_count": int(prepared["all_query_count"]),
        "error_query_count": int(prepared["error_query_count"]),
        "public_text_policy": (
            "Only IDs, hashes, numeric audit fields, and generated ranking "
            "metadata are included; source prompts and responses are omitted."
        ),
        "legacy_local_metadata_excluded": [
            {
                "path": (
                    "results/research_public_gsm8k_v1/"
                    "data_manifest_full_v2_fuzzy/data_manifest.json"
                ),
                "reason": (
                    "Legacy local metadata contains a machine-specific path "
                    "and is not an input binding of the full-pool scoring run."
                ),
            }
        ],
    }
    data_provenance["data_provenance_sha256"] = canonical_json_sha256(
        data_provenance
    )

    chunk_manifest_sources: list[tuple[Path, Path]] = []
    chunk_summary: dict[str, Any] = {}
    all_chunk_manifests: list[dict[str, Any]] = []
    for kind in ("query", "candidate"):
        sources = sorted((run_dir / "embedding_chunks" / kind).glob("chunk_*.json"))
        expected = 7 if kind == "query" else 67
        if len(sources) != expected:
            raise ValueError(f"{kind} chunk manifest count changed")
        rows = 0
        for source in sources:
            manifest = _read_json(source)
            rows += int(manifest["row_count"])
            all_chunk_manifests.append(manifest)
            destination = (
                Path("embedding_chunk_manifests") / kind / source.name
            )
            chunk_manifest_sources.append((source, destination))
        chunk_summary[kind] = {
            "chunk_count": len(sources),
            "row_count": rows,
            "ordered_manifest_file_sha256": [
                file_sha256(path) for path in sources
            ],
        }

    scoring_summary = {
        "schema_version": "rds-public-scoring-summary-v1",
        "source_run_contract_sha256": local_contract["run_contract_sha256"],
        "prepared_sha256": file_sha256(prepared_path),
        "finalization_manifest_sha256": file_sha256(finalization_path),
        "candidate_inventory_sha256": prepared["candidate_inventory"]["sha256"],
        "query_inventory_sha256": prepared["query_inventory"]["sha256"],
        "candidate_scores_sha256": file_sha256(scores_source),
        "metrics_sha256": file_sha256(metrics_source),
        "query_embeddings_tensor_sha256": metrics[
            "query_embeddings_sha256"
        ],
        "candidate_embeddings_tensor_sha256": metrics[
            "candidate_embeddings_sha256"
        ],
        "chunk_summary": chunk_summary,
        "maximum_observed_temperature_c": max(
            int(manifest["maximum_observed_temperature_c"])
            for manifest in all_chunk_manifests
        ),
        "maximum_peak_memory_gib": max(
            float(manifest["peak_memory_gib"])
            for manifest in all_chunk_manifests
        ),
        "rank_metrics": {
            "all_vs_error_rank_spearman": metrics[
                "all_vs_error_rank_spearman"
            ],
            "all_vs_error_top_budget_jaccard": metrics[
                "all_vs_error_top_budget_jaccard"
            ],
            "selection_budget": metrics["selection_budget"],
        },
        "selections": {
            strategy: {
                "source_path": _repo_relative(selection_sources[strategy]),
                "file_sha256": file_sha256(selection_sources[strategy]),
                "selected_id_sha256": selections[strategy][
                    "selected_id_sha256"
                ],
                "selected_source_counts": selections[strategy][
                    "selected_source_counts"
                ],
                "selected_total_tokens": selections[strategy][
                    "selected_total_tokens"
                ],
                "selected_supervised_tokens": selections[strategy][
                    "selected_supervised_tokens"
                ],
            }
            for strategy in ("rds_all", "rds_error")
        },
        "local_only_tensor_policy": (
            "The 74 tensor files remain local and immutable. Their file and "
            "tensor hashes are preserved by the copied chunk manifests and "
            "were independently deep-audited before package creation."
        ),
        "claim_boundary": local_contract["claim_boundary"],
    }
    scoring_summary["scoring_summary_sha256"] = canonical_json_sha256(
        scoring_summary
    )

    copy_plan: list[tuple[Path, Path]] = [
        (prepared_path, Path("artifacts/prepared.json")),
        (
            run_dir / prepared["candidate_inventory"]["path"],
            Path("artifacts/candidate_inventory.jsonl"),
        ),
        (
            run_dir / prepared["query_inventory"]["path"],
            Path("artifacts/query_inventory.jsonl"),
        ),
        (finalization_path, Path("artifacts/finalization_manifest.json")),
        (scores_source, Path("artifacts/candidate_scores.jsonl")),
        (metrics_source, Path("artifacts/metrics.json")),
        (
            selection_sources["rds_all"],
            Path("selections/rds_all_selection_manifest.json"),
        ),
        (
            selection_sources["rds_error"],
            Path("selections/rds_error_selection_manifest.json"),
        ),
        *chunk_manifest_sources,
    ]
    text_payloads = {
        "public_run_contract.json": json.dumps(
            public_contract,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "data_provenance.json": json.dumps(
            data_provenance,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "scoring_summary.json": json.dumps(
            scoring_summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    }
    findings: list[dict[str, str]] = []
    for label, text in text_payloads.items():
        findings.extend(_scan_text(label=label, text=text))
    for source, destination in copy_plan:
        text = source.read_text(encoding="utf-8")
        findings.extend(_scan_text(label=destination.as_posix(), text=text))
    if findings:
        raise ValueError(f"public evidence pre-scan failed: {findings}")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "public_run_contract.json", public_contract)
    _write_json(output_dir / "data_provenance.json", data_provenance)
    _write_json(output_dir / "scoring_summary.json", scoring_summary)
    for source, destination in copy_plan:
        _copy_exclusive(source, output_dir / destination)

    included_files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            included_files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": PUBLIC_SCHEMA,
        "status": "COMPLETE",
        "included_files": included_files,
        "included_file_count": len(included_files),
        "privacy_scan": {
            "status": "PASS",
            "windows_absolute_path_hits": 0,
            "unix_private_path_hits": 0,
            "secret_like_hits": 0,
            "raw_source_text_field_hits": 0,
        },
        "immutable_local_evidence": {
            "changed": False,
            "source_run_contract_path": _repo_relative(contract_path),
            "source_run_contract_file_sha256": file_sha256(contract_path),
            "source_run_contract_self_sha256": local_contract[
                "run_contract_sha256"
            ],
        },
        "excluded_from_public_package": [
            "machine-specific local run contract",
            "legacy local data-manifest metadata",
            "74 local tensor files",
            "model weights and datasets",
        ],
        "claim_boundary": local_contract["claim_boundary"],
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    _write_json(output_dir / "PUBLIC_EVIDENCE_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "included_file_count": manifest["included_file_count"],
                "manifest_content_sha256": manifest[
                    "manifest_content_sha256"
                ],
                "privacy_scan": manifest["privacy_scan"],
                "selection_file_sha256": {
                    strategy: file_sha256(selection_sources[strategy])
                    for strategy in ("rds_all", "rds_error")
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
