"""Create a new frozen 16-cell matrix config from audited selection manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.experiment.budget_equivalent_protocol import (  # noqa: E402
    phase1_jobs,
    read_json_object,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/budget_equivalent_v3_protocol_frozen.json"),
    )
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    protocol_config = read_json_object(args.protocol_config.resolve())
    selection_root = args.selection_root.resolve()
    index_path = selection_root / "selection_index.json"
    gates_path = selection_root / "information_gates.json"
    index = read_json_object(index_path)
    gates = read_json_object(gates_path)
    if index.get("selection_count") != 16 or index.get("formal_phase1_ready") is not True:
        raise ValueError("selection index is not formal Phase 1 ready")
    if gates.get("targeted_policy_gate_passed") is not True:
        raise ValueError("targeted-policy gate failed")
    indexed = {
        (int(row["replicate_index"]), str(row["method"])): row
        for row in index["selections"]
    }
    jobs = []
    for job in phase1_jobs(protocol_config):
        binding = indexed[(int(job["replicate_index"]), str(job["method"]))]
        manifest_path = selection_root / binding["path"]
        jobs.append(
            job
            | {
                "selection_manifest": {
                    "path": _relative(manifest_path),
                    "sha256": file_sha256(manifest_path),
                }
            }
        )
    payload = {
        "phase1_protocol_version": "budget-equivalent-phase1-matrix-v3",
        "protocol_config": {
            "path": "configs/public_gsm8k_v1.json",
            "sha256": file_sha256(ROOT / "configs/public_gsm8k_v1.json"),
        },
        "base_recipe_config": {
            "path": "configs/budget_equivalent_lora_v3.json",
            "sha256": file_sha256(ROOT / "configs/budget_equivalent_lora_v3.json"),
        },
        "data_manifest": {
            "directory": "results/research_public_gsm8k_v1/data_manifest_full_v2_fuzzy",
            "required_files": {
                "gsm8k_records.jsonl": file_sha256(
                    ROOT
                    / "results/research_public_gsm8k_v1/data_manifest_full_v2_fuzzy/"
                    "gsm8k_records.jsonl"
                ),
                "tulu_candidate_pool.jsonl": file_sha256(
                    ROOT
                    / "results/research_public_gsm8k_v1/data_manifest_full_v2_fuzzy/"
                    "tulu_candidate_pool.jsonl"
                ),
            },
        },
        "selection_index": {"path": _relative(index_path), "sha256": file_sha256(index_path)},
        "information_gates": {"path": _relative(gates_path), "sha256": file_sha256(gates_path)},
        "methods": list(protocol_config["methods"]),
        "job_order": jobs,
        "output_root": ".aris/compute/budget_equivalent_phase1_runs_v3",
        "training": {
            "selection_budget": 500,
            "epochs": 2,
            "optimizer_steps": 64,
            "max_length": 512,
            "micro_batch_size": 1,
            "loss_normalization": "optimizer_step_response_token_sum_over_count",
            "checkpoint_every_optimizer_steps": 8,
            "attention_implementation": "sdpa",
            "gradient_checkpointing": False,
            "single_training_process": True
        },
        "evaluation": {
            "split": "held_out_test",
            "expected_record_count": 1319,
            "worker_count": 2,
            "physical_batch_size_per_worker": 1,
            "forbid_batch_size_above_one": True,
            "cuda_device_index": 0,
            "shards": [
                {"shard_id": "test_shard0", "start_index": 0, "end_index": 660},
                {"shard_id": "test_shard1", "start_index": 660, "end_index": 1319}
            ],
            "max_input_length": 512,
            "max_new_tokens": 256,
            "do_sample": False,
            "num_beams": 1,
            "padding_side": "left",
            "prompt_version": "gsm8k_base_completion_v2_one_shot_frozen",
            "parser_policy": "strict_final_marker_then_last_numeric_fallback"
        },
        "resources": {
            "min_free_system_memory_gib": 16,
            "min_free_disk_gib": 50,
            "max_training_peak_allocated_gib": 20,
            "max_worker_peak_allocated_gib": 13,
            "hard_stop_temperature_c": 80
        },
        "execution_policy": {
            "one_cell_per_invocation": True,
            "automatic_next_cell": False,
            "require_clean_repo_before_new_run": True,
            "runtime_output_is_git_ignored": True,
            "stdout_withholds_accuracy_and_method_comparison": True,
            "accuracy_blind_until_all_audits": True,
            "required_audited_cells_before_unblinding": 16
        }
    }
    with args.output_config.resolve().open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(file_sha256(args.output_config.resolve()))


if __name__ == "__main__":
    main()
