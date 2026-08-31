"""Open and analyze Phase 1A only after all 16 formal and OOD audits pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_blind_aggregate import (  # noqa: E402
    guarded_blind_aggregation,
)
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    phase1_registry,
    read_json_object,
)
from eg_sft.experiment.budget_equivalent_phase1_analysis import (  # noqa: E402
    TASKS,
    cell_metrics,
    comparison_report,
    read_jsonl,
    require_all_audited,
    summarize_methods,
    validate_task_alignment,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


DEFAULT_CONFIG = Path("configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json")
DEFAULT_PRIVATE_MAP = Path(
    ".aris/control/budget_equivalent_phase1_blind_v2/private_blind_map.json"
)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing SHA-256 sidecar: {path}")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    if file_sha256(path) != expected:
        raise ValueError(f"artifact hash changed: {path}")


def _read_audited_cells(
    *, config: dict[str, Any], registry: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_cell = {str(row["cell_id"]): row for row in registry["jobs"]}
    cells = []
    evidence = []
    for job in config["job_order"]:
        cell_id = str(job["cell_id"])
        registered = by_cell[cell_id]
        run_dirs = list(registered["run_dirs"])
        if len(run_dirs) != 1:
            raise ValueError(f"expected exactly one run directory for {cell_id}")
        run_dir = Path(run_dirs[0]).resolve()
        formal = run_dir / "audit" / "formal_cell_audit.json"
        ood = run_dir / "audit" / "ood_audit.json"
        _verify_sidecar(formal)
        _verify_sidecar(ood)
        if read_json_object(formal).get("status") != "PASS":
            raise ValueError(f"formal audit changed: {cell_id}")
        if read_json_object(ood).get("status") != "PASS":
            raise ValueError(f"OOD audit changed: {cell_id}")

        paths = {
            "gsm8k": run_dir / "evaluation" / "merged" / "raw_outputs.jsonl",
            "svamp": run_dir / "evaluation" / "ood" / "svamp" / "merged" / "raw_outputs.jsonl",
            "asdiv_numeric": run_dir
            / "evaluation"
            / "ood"
            / "asdiv_numeric"
            / "merged"
            / "raw_outputs.jsonl",
            "multiarith": run_dir
            / "evaluation"
            / "ood"
            / "multiarith"
            / "merged"
            / "raw_outputs.jsonl",
        }
        tasks = {task: read_jsonl(path) for task, path in paths.items()}
        training_metrics = read_json_object(
            run_dir / "training_complete" / "training_metrics.json"
        )
        cells.append(
            {
                "cell_id": cell_id,
                "method": str(job["method"]),
                "replicate_index": int(job["replicate_index"]),
                "train_seed": int(job["train_seed"]),
                "tasks": tasks,
                "training": {
                    key: training_metrics[key]
                    for key in (
                        "supervised_tokens_seen",
                        "optimizer_steps_completed",
                        "training_wall_seconds",
                        "peak_training_memory_gib",
                        "adapter_model_sha256",
                    )
                },
            }
        )
        evidence.append(
            {
                "cell_id": cell_id,
                "run_id": read_json_object(run_dir / "manifest.json")["run_id"],
                "formal_audit_sha256": file_sha256(formal),
                "ood_audit_sha256": file_sha256(ood),
                "raw_output_sha256": {
                    task: file_sha256(path) for task, path in paths.items()
                },
                "tokenizer_json_sha256": file_sha256(
                    run_dir / "training_complete" / "tokenizer" / "tokenizer.json"
                ),
            }
        )
    return cells, evidence


def _claim_boundary_markdown(
    *, method_summary: dict[str, Any], comparisons: dict[str, Any]
) -> str:
    lines = [
        "# Phase 1A 解盲结果与结论边界",
        "",
        "## 单格原始汇总",
        "",
        "| 方法 | 选择名单数 | GSM8K准确率均值±名单标准差 | OOD宏平均均值±名单标准差 |",
        "|---|---:|---:|---:|",
    ]
    for method, row in method_summary.items():
        gsm = row["tasks"]["gsm8k"]["accuracy"]
        ood = row["tasks"]["ood_macro"]["accuracy"]
        lines.append(
            f"| `{method}` | {row['selection_replicate_count']} | "
            f"{gsm['mean']:.4f} ± {gsm['selection_replicate_sd']:.4f} | "
            f"{ood['mean']:.4f} ± {ood['selection_replicate_sd']:.4f} |"
        )
    lines.extend(["", "## 预注册比较", ""])
    for name, row in comparisons["comparisons"].items():
        gsm = row["accuracy"]["tasks"]["gsm8k"]
        lines.append(
            f"- `{name}`：GSM8K差异 {gsm['point_difference']:+.4f}，"
            f"95% CI [{gsm['ci95'][0]:+.4f}, {gsm['ci95'][1]:+.4f}]，"
            f"诊断为 `{row['primary_gsm8k_threshold_diagnostic']}`。"
        )
    lines.extend(
        [
            "",
            "## 允许与不允许的结论",
            "",
            "- 允许：报告四个独立选择名单下的初步方向、名单方差、GSM8K与OOD差异。",
            "- 不允许：仅凭Phase 1A声称方法最终有效、最终等效或最终无效。",
            "- 原因：每个选择名单目前只有一个训练随机种子，训练方差尚未独立估计。",
            "- 下一门槛：按冻结Phase 2设计增加训练种子与选择名单后，才能形成确认性结论。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--private-map", type=Path, default=DEFAULT_PRIVATE_MAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()

    config_path = args.config.resolve()
    private_map_path = args.private_map.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json_object(config_path)
    registry = phase1_registry(repo_root=ROOT, config_path=config_path)
    require_all_audited(registry)

    private_map = read_json_object(private_map_path)
    if private_map.get("matrix_sha256") != file_sha256(config_path):
        raise ValueError("private blind map is not bound to the frozen matrix")
    ood_status = {
        str(row["cell_id"]): (
            "AUDITED_PASS" if row["status"] == "AUDITED_PASS" else "PENDING"
        )
        for row in registry["jobs"]
    }
    gate = guarded_blind_aggregation(
        private_map=private_map,
        registry=registry,
        ood_status_by_cell=ood_status,
        ood_required=True,
    )
    if gate.get("unblinding_permitted") is not True:
        raise ValueError("formal unblinding gate remains closed")

    cells, evidence = _read_audited_cells(config=config, registry=registry)
    validate_task_alignment(cells)
    tokenizer_hashes = {str(row["tokenizer_json_sha256"]) for row in evidence}
    if len(tokenizer_hashes) != 1:
        raise ValueError("tokenizer artifacts differ across Phase 1 cells")
    first_run = Path(registry["jobs"][0]["run_dirs"][0]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        first_run / "training_complete" / "tokenizer", use_fast=True
    )

    def token_length(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    cell_rows = [cell_metrics(cell, token_length=token_length) for cell in cells]
    method_summary = summarize_methods(cell_rows)
    comparisons = comparison_report(
        cells=cells,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    cell_path = output_dir / "cell_metrics.jsonl"
    with cell_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in cell_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = output_dir / "method_summary.json"
    comparisons_path = output_dir / "comparisons.json"
    gate_path = output_dir / "unblinding_gate.json"
    _write_json_exclusive(summary_path, method_summary)
    _write_json_exclusive(comparisons_path, comparisons)
    _write_json_exclusive(gate_path, gate)
    claim_path = output_dir / "claim_boundary_cn.md"
    claim_path.write_text(
        _claim_boundary_markdown(
            method_summary=method_summary,
            comparisons=comparisons,
        ),
        encoding="utf-8",
        newline="\n",
    )
    outputs = [cell_path, summary_path, comparisons_path, gate_path, claim_path]
    manifest = {
        "schema_version": "budget-equivalent-phase1a-analysis-manifest-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "matrix_config_sha256": file_sha256(config_path),
        "private_map_sha256": file_sha256(private_map_path),
        "audited_cell_count": 16,
        "evaluation_tasks": list(TASKS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "cell_evidence": evidence,
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
        "phase1a_is_directional_pilot_not_confirmatory": True,
    }
    manifest_path = output_dir / "analysis_manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    digest = hashlib.sha256()
    for path in sorted((*outputs, manifest_path), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(file_sha256(path).encode("ascii") + b"\n")
    (output_dir / "ANALYSIS_COMPLETE.sha256").write_text(
        f"{digest.hexdigest()}  phase1a_analysis_outputs\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "budget_equivalent_phase1a_unblinded_analysis",
                "audited_cell_count": 16,
                "output_dir": str(output_dir),
                "phase1a_is_directional_pilot_not_confirmatory": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
