"""CPU-only preflight for every frozen Phase-2 v7 cell and deployment asset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    validate_reference_manifest,
    write_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    resolve_phase1_contract,
)
from eg_sft.experiment.budget_equivalent_ood_runtime import (  # noqa: E402
    OOD_DATASETS,
    resolve_ood_contract,
)
from eg_sft.experiment.phase2_crossed_v7 import (  # noqa: E402
    validate_phase2_matrix,
)
from eg_sft.experiment.phase2_v7_control import (  # noqa: E402
    Phase2StateStore,
    worker_schedule,
)


def _binding(repo_root: Path, value: dict[str, Any], label: str) -> Path:
    path = (repo_root / str(value["path"])).resolve()
    path.relative_to(repo_root.resolve())
    if not path.is_file() or file_sha256(path) != str(value["sha256"]):
        raise ValueError(f"{label} binding changed")
    return path


def preflight(
    *, repo_root: Path, config_path: Path, output_root: Path, min_free_gib: float
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    matrix = read_json(config_path)
    validate_phase2_matrix(matrix)
    matrix_sha = file_sha256(config_path)

    resolved_cells = []
    for row in matrix["job_order"]:
        contract = resolve_phase1_contract(
            repo_root=repo_root,
            config_path=config_path,
            cell_id=str(row["cell_id"]),
        )
        if (
            contract["selection"]["file_sha256"]
            != row["parent_selection_manifest_sha256"]
            or len(contract["selection"]["selected"]) != 500
        ):
            raise ValueError(f"selection contract changed: {row['cell_id']}")
        resolved_cells.append(
            {
                "cell_id": row["cell_id"],
                "parent_cell_id": row["parent_cell_id"],
                "method": row["method"],
                "replicate_index": row["replicate_index"],
                "train_seed": row["train_seed"],
                "selection_manifest_sha256": contract["selection"]["file_sha256"],
                "selected_id_sha256": contract["selection"]["selected_id_sha256"],
                "selected_count": len(contract["selection"]["selected"]),
            }
        )
    ood = {}
    for dataset in OOD_DATASETS:
        contract = resolve_ood_contract(
            repo_root=repo_root,
            matrix_config_path=config_path,
            dataset=dataset,
        )
        ood[dataset] = {
            "record_count": len(contract["records"]),
            "manifest_sha256": contract["manifest_sha256"],
            "records_sha256": contract["records_sha256"],
        }

    legacy_path = _binding(
        repo_root,
        matrix["backend_contracts"]["legacy_batch1"],
        "legacy batch1 contract",
    )
    legacy = read_json(legacy_path)
    for role in ("base_model_16",):
        binding = legacy[role]
        reference = (repo_root / binding["reference_path"]).resolve()
        manifest_path = (repo_root / binding["manifest_path"]).resolve()
        if (
            file_sha256(reference) != binding["reference_sha256"]
            or file_sha256(manifest_path) != binding["manifest_sha256"]
        ):
            raise ValueError(f"{role} reference binding changed")
        validate_reference_manifest(
            manifest=read_json(manifest_path), reference_path=reference
        )
    adapter = legacy["archived_adapter_16"]
    semantic_reference = (
        repo_root / adapter["historical_semantic_reference_path"]
    ).resolve()
    semantic_manifest = (repo_root / adapter["historical_manifest_path"]).resolve()
    if (
        file_sha256(semantic_reference)
        != adapter["historical_semantic_reference_sha256"]
        or file_sha256(semantic_manifest) != adapter["historical_manifest_sha256"]
    ):
        raise ValueError("archived adapter semantic reference changed")
    adapter_files = sorted(repo_root.glob("review_pack/**/adapter_model.safetensors"))
    matching_adapters = [
        path for path in adapter_files if file_sha256(path) == adapter["adapter_model_sha256"]
    ]
    if len(matching_adapters) != 1:
        raise ValueError("exactly one frozen archived adapter must be present")

    free_gib = shutil.disk_usage(repo_root).free / 1024**3
    if free_gib < min_free_gib:
        raise ValueError(
            f"insufficient CPU-preflight disk: {free_gib:.2f} GiB < {min_free_gib:.2f} GiB"
        )
    control = Phase2StateStore(
        root=output_root / "control_template", matrix_path=config_path
    )
    control_report = control.initialize()
    registry = control.registry()
    command_rows = []
    for worker_id in ("gpu0", "gpu1"):
        for index, cell_id in enumerate(worker_schedule(matrix, worker_id), start=1):
            command_rows.append(
                {
                    "worker_id": worker_id,
                    "worker_position": index,
                    "cell_id": cell_id,
                    "contract_only_command": (
                        "python scripts/run_identifiable_budget_v4_cell.py "
                        f"--config configs/{config_path.name} --cell-id {cell_id} "
                        "--contract-only"
                    ),
                }
            )
    report = {
        "schema_version": "phase2-v7-cpu-preflight-v1",
        "status": "PASS",
        "matrix_sha256": matrix_sha,
        "resolved_cell_count": len(resolved_cells),
        "resolved_cells": resolved_cells,
        "ood": ood,
        "archived_adapter_sha256": adapter["adapter_model_sha256"],
        "archived_adapter_relative_path": matching_adapters[0]
        .resolve()
        .relative_to(repo_root)
        .as_posix(),
        "control_initialization": control_report,
        "control_registry_state_counts": registry["state_counts"],
        "command_count": len(command_rows),
        "free_disk_gib": free_gib,
        "gpu_accessed": False,
        "accuracy_withheld": True,
    }
    write_exclusive_or_verify(
        output_root / "preflight_report.json", canonical_json_bytes(report)
    )
    command_bytes = b"".join(canonical_json_bytes(row) for row in command_rows)
    write_exclusive_or_verify(output_root / "commands.jsonl", command_bytes)
    write_exclusive_or_verify(
        output_root / "control_registry.json", canonical_json_bytes(registry)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_crossed_48cell_v7.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/phase2_v7_preflight"),
    )
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    args = parser.parse_args()
    report = preflight(
        repo_root=ROOT,
        config_path=args.config,
        output_root=args.output_root,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
