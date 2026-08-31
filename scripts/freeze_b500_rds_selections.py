"""Freeze B=500 all-query and error-query RDS+ selection manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.rds_full_pool import (  # noqa: E402
    build_b500_selection_manifest,
    canonical_json_sha256,
)
from eg_sft.selection.query_groups import load_jsonl  # noqa: E402
from eg_sft.training.b500 import file_sha256  # noqa: E402


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


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside repository: {path}") from error


def _resolve_run_artifact(
    *,
    run_dir: Path,
    binding: dict[str, Any],
) -> Path:
    relative = Path(str(binding["path"]))
    if relative.is_absolute():
        raise ValueError("finalized artifact path must be relative")
    resolved = (run_dir / relative).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError("finalized artifact escapes the run directory") from error
    if file_sha256(resolved) != binding["sha256"]:
        raise ValueError(f"finalized artifact hash changed: {relative.as_posix()}")
    return resolved


def _verify_manifest_self_hash(manifest: dict[str, Any]) -> None:
    claimed = manifest.get("manifest_content_sha256")
    payload = dict(manifest)
    payload.pop("manifest_content_sha256", None)
    if claimed != canonical_json_sha256(payload):
        raise ValueError("selection manifest content hash changed")


def _validate_existing(
    *,
    path: Path,
    strategy: str,
    budget: int,
    selection_seed: int,
) -> dict[str, Any]:
    manifest = _read_json(path)
    _verify_manifest_self_hash(manifest)
    if manifest.get("strategy") != strategy:
        raise ValueError(f"existing {strategy} strategy changed")
    if int(manifest.get("budget", -1)) != budget:
        raise ValueError(f"existing {strategy} budget changed")
    if int(manifest.get("selection_seed", -1)) != selection_seed:
        raise ValueError(f"existing {strategy} selection seed changed")
    if len(manifest.get("selected_candidates", [])) != budget:
        raise ValueError(f"existing {strategy} selected count changed")
    return manifest


def _validate_scores_against_inventory(
    *,
    scores: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> None:
    eligible_rows = [
        row for row in inventory if bool(row["response_only_trainable"])
    ]
    eligible = {
        str(row["candidate_id"]): row
        for row in eligible_rows
    }
    if len(scores) != len(eligible):
        raise ValueError("candidate score count differs from eligible inventory")
    if len(eligible) != len(eligible_rows):
        raise ValueError("eligible inventory has duplicate candidate IDs")
    frozen_fields = (
        "candidate_id",
        "source_dataset",
        "source_id",
        "source_index",
        "prompt_sha256",
        "response_sha256",
        "user_prompt_sha256",
        "selection_priority_sha256",
        "selection_rank",
        "candidate_order_index",
        "eligible_index",
        "rds_text_sha256",
        "total_tokens",
        "supervised_tokens",
    )
    for row in scores:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in eligible:
            raise ValueError(f"score row is outside eligible pool: {candidate_id}")
        frozen = eligible[candidate_id]
        for field in frozen_fields:
            if row.get(field) != frozen.get(field):
                raise ValueError(
                    f"{candidate_id} changed frozen inventory field {field}"
                )
    for rank_field in ("all_query_rank", "error_query_rank"):
        ranks = sorted(int(row[rank_field]) for row in scores)
        if ranks != list(range(len(scores))):
            raise ValueError(f"{rank_field} is not a complete zero-based order")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rds-all-output", type=Path, required=True)
    parser.add_argument("--rds-error-output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    contract_path = run_dir / "run_contract.json"
    prepared_path = run_dir / "prepared.json"
    finalization_path = run_dir / "finalization_manifest.json"
    contract = _read_json(contract_path)
    prepared = _read_json(prepared_path)
    finalization = _read_json(finalization_path)
    if contract.get("scope") != "formal_10000_candidate_pool":
        raise ValueError("B=500 manifests require the formal 10,000-candidate scope")
    claimed_contract_hash = contract.get("run_contract_sha256")
    contract_payload = dict(contract)
    contract_payload.pop("run_contract_sha256", None)
    if claimed_contract_hash != canonical_json_sha256(contract_payload):
        raise ValueError("run contract self-hash changed")
    if prepared.get("run_contract_sha256") != claimed_contract_hash:
        raise ValueError("prepared artifact belongs to another run contract")
    if finalization.get("run_contract_sha256") != claimed_contract_hash:
        raise ValueError("finalization belongs to another run contract")
    if int(prepared["audited_candidate_count"]) != 10000:
        raise ValueError("formal candidate audit is not exactly 10,000 rows")
    if int(prepared["all_query_count"]) != 448:
        raise ValueError("formal all-query group is not exactly 448 rows")
    if int(prepared["error_query_count"]) != 99:
        raise ValueError("formal error-query group is not exactly 99 rows")

    scores_binding = finalization["artifacts"]["candidate_scores"]
    metrics_binding = finalization["artifacts"]["metrics"]
    scores_path = _resolve_run_artifact(
        run_dir=run_dir,
        binding=scores_binding,
    )
    metrics_path = _resolve_run_artifact(
        run_dir=run_dir,
        binding=metrics_binding,
    )
    scores = load_jsonl(scores_path)
    inventory_path = run_dir / str(prepared["candidate_inventory"]["path"])
    if file_sha256(inventory_path) != prepared["candidate_inventory"]["sha256"]:
        raise ValueError("candidate inventory hash changed")
    inventory = load_jsonl(inventory_path)
    _validate_scores_against_inventory(scores=scores, inventory=inventory)
    metrics = _read_json(metrics_path)
    if metrics.get("candidate_scores_sha256") != file_sha256(scores_path):
        raise ValueError("metrics do not bind the candidate score file")
    if int(metrics.get("selection_budget", -1)) != 500:
        raise ValueError("finalized scoring budget changed")

    budget = int(contract["selection"]["budget"])
    selection_seed = int(contract["selection"]["selection_seed"])
    if budget != 500:
        raise ValueError("selection budget changed")
    provenance = {
        "run_contract_path": _repo_relative(contract_path),
        "run_contract_sha256": claimed_contract_hash,
        "prepared_path": _repo_relative(prepared_path),
        "prepared_sha256": file_sha256(prepared_path),
        "candidate_inventory_sha256": file_sha256(inventory_path),
        "finalization_manifest_path": _repo_relative(finalization_path),
        "finalization_manifest_sha256": file_sha256(finalization_path),
        "candidate_scores_path": _repo_relative(scores_path),
        "candidate_scores_sha256": file_sha256(scores_path),
        "metrics_path": _repo_relative(metrics_path),
        "metrics_sha256": file_sha256(metrics_path),
        "representation_version": metrics["representation_version"],
        "audited_candidate_count": int(metrics["audited_candidate_count"]),
        "eligible_candidate_count": int(metrics["eligible_candidate_count"]),
        "all_query_count": int(metrics["all_query_count"]),
        "error_query_count": int(metrics["error_query_count"]),
    }
    outputs = {
        "rds_all": args.rds_all_output.resolve(),
        "rds_error": args.rds_error_output.resolve(),
    }
    report: dict[str, Any] = {}
    for strategy, output in outputs.items():
        if output.is_file():
            manifest = _validate_existing(
                path=output,
                strategy=strategy,
                budget=budget,
                selection_seed=selection_seed,
            )
            status = "ALREADY_COMPLETE"
        else:
            manifest = build_b500_selection_manifest(
                strategy=strategy,
                score_rows=scores,
                budget=budget,
                selection_seed=selection_seed,
                scoring_provenance=provenance,
            )
            _write_json(output, manifest)
            status = "CREATED"
        report[strategy] = {
            "status": status,
            "path": str(output),
            "file_sha256": file_sha256(output),
            "selected_id_sha256": manifest["selected_id_sha256"],
            "selected_source_counts": manifest["selected_source_counts"],
            "selected_total_tokens": manifest["selected_total_tokens"],
            "selected_supervised_tokens": manifest["selected_supervised_tokens"],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
