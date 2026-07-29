"""Audit one completed random-B500 adapter evaluation and reload the adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    PROMPT_VERSION,
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.experiment.b500_engineering_audit import (  # noqa: E402
    audit_completed_evaluation,
    summarize_adapter_tensors,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _committed_file_sha256(relative_path: Path) -> str | None:
    process = subprocess.run(
        ["git", "show", f"HEAD:{relative_path.as_posix()}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        return None
    return hashlib.sha256(process.stdout).hexdigest()


def _audit_code_provenance() -> dict[str, Any]:
    paths = [
        Path("scripts/audit_b500_engineering_run.py"),
        Path("src/eg_sft/experiment/b500_engineering_audit.py"),
        Path("tests/test_b500_engineering_audit.py"),
    ]
    files: dict[str, Any] = {}
    all_match = True
    for relative_path in paths:
        working_sha256 = file_sha256(ROOT / relative_path)
        committed_sha256 = _committed_file_sha256(relative_path)
        matches = working_sha256 == committed_sha256
        all_match = all_match and matches
        files[relative_path.as_posix()] = {
            "working_tree_sha256": working_sha256,
            "committed_sha256": committed_sha256,
            "working_tree_matches_commit": matches,
        }
    if not all_match:
        raise ValueError(
            "audit implementation must exactly match the recorded git commit"
        )
    return {
        "git_commit": _git_commit(),
        "command": [sys.executable, *sys.argv],
        "files": files,
        "all_implementation_files_match_commit": True,
    }


def _load_and_compare_adapter_logits(
    *,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    run_dir: Path,
    first_question: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("the independent adapter reload check requires CUDA")
    model_config = protocol["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir / "tokenizer",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        run_dir / "adapter",
        is_trainable=False,
    ).to("cuda")
    model.config.use_cache = True
    model.eval()

    active_adapters = list(model.active_adapters)
    if not active_adapters:
        raise ValueError("no adapter is active after PeftModel.from_pretrained")
    prompt = build_evaluation_prompt(first_question)
    encoded = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(recipe["evaluation"]["max_input_length"]),
    ).to("cuda")
    with torch.inference_mode():
        active_logits = model(**encoded).logits[:, -1, :].float().cpu()
        with model.disable_adapter():
            base_logits = model(**encoded).logits[:, -1, :].float().cpu()
    max_abs_logit_difference = float(
        torch.max(torch.abs(active_logits - base_logits)).item()
    )
    if max_abs_logit_difference <= 0:
        raise ValueError("active adapter does not change the fixed-prompt logits")

    adapter_named_parameters = [
        name for name, _ in model.named_parameters() if "lora_" in name
    ]
    if not adapter_named_parameters:
        raise ValueError("reloaded model exposes no LoRA parameters")
    return {
        "status": "PASS",
        "active_adapters": active_adapters,
        "adapter_parameter_tensor_count_in_model": len(
            adapter_named_parameters
        ),
        "fixed_prompt_input_tokens": int(encoded["input_ids"].shape[1]),
        "active_vs_disabled_max_abs_logit_difference": (
            max_abs_logit_difference
        ),
        "adapter_changes_fixed_prompt_logits": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--recipe-config", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-directory-name", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    protocol_path = args.protocol_config.resolve()
    recipe_path = args.recipe_config.resolve()
    execution_path = args.execution_config.resolve()
    data_manifest_dir = args.data_manifest_dir.resolve()
    run_dir = args.run_dir.resolve()
    evaluation_dir = run_dir / args.evaluation_directory_name
    output_path = (
        args.output.resolve()
        if args.output is not None
        else evaluation_dir / "engineering_audit.json"
    )
    if output_path.exists():
        raise FileExistsError(f"audit output already exists: {output_path}")

    protocol = _read_json(protocol_path)
    recipe = _read_json(recipe_path)
    run_manifest = _read_json(run_dir / "manifest.json")
    evaluation_manifest = _read_json(evaluation_dir / "manifest.json")
    metrics_path = evaluation_dir / "metrics.json"
    raw_outputs_path = evaluation_dir / "raw_outputs.jsonl"
    metrics = _read_json(metrics_path)
    rows = read_jsonl(raw_outputs_path)
    frozen_records = [
        row
        for row in read_jsonl(data_manifest_dir / "gsm8k_records.jsonl")
        if row["protocol_split"] == recipe["evaluation"]["split"]
    ]
    frozen_records.sort(
        key=lambda row: (int(row["source_index"]), str(row["record_id"]))
    )

    row_audit = audit_completed_evaluation(
        rows=rows,
        frozen_records=frozen_records,
        metrics=metrics,
        prompt_version=PROMPT_VERSION,
    )
    gsm_config = protocol["datasets"]["gsm8k"]
    gsm_test = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        split="test",
        revision=gsm_config["revision"],
    )
    if len(gsm_test) != len(frozen_records):
        raise ValueError("pinned GSM8K test count changed")
    for index, (source_row, frozen, saved) in enumerate(
        zip(gsm_test, frozen_records, rows, strict=True)
    ):
        validate_gsm8k_source_row(frozen, source_row)
        rescored = score_generation(
            record=frozen,
            gold_answer_text=source_row["answer"],
            generated_text=saved["raw_output"],
        )
        if rescored != saved:
            raise ValueError(f"saved row does not rescore exactly at {index}")

    adapter_path = run_dir / "adapter" / "adapter_model.safetensors"
    raw_outputs_sha256 = file_sha256(raw_outputs_path)
    adapter_model_sha256 = file_sha256(adapter_path)
    if raw_outputs_sha256 != metrics["raw_outputs_sha256"]:
        raise ValueError("raw-output file hash does not match metrics")
    if len(
        {
            adapter_model_sha256,
            metrics["adapter_model_sha256"],
            evaluation_manifest["adapter_model_sha256"],
        }
    ) != 1:
        raise ValueError("adapter hash contract is inconsistent")
    config_hashes = {
        "protocol_config_sha256": file_sha256(protocol_path),
        "recipe_config_sha256": file_sha256(recipe_path),
        "execution_policy_sha256": file_sha256(execution_path),
    }
    for key, value in config_hashes.items():
        if evaluation_manifest[key] != value:
            raise ValueError(f"evaluation manifest mismatch for {key}")

    serialized_adapter = summarize_adapter_tensors(load_file(adapter_path))
    reload_audit = _load_and_compare_adapter_logits(
        protocol=protocol,
        recipe=recipe,
        run_dir=run_dir,
        first_question=str(gsm_test[0]["question"]),
    )
    audit_code_provenance = _audit_code_provenance()
    provenance_dir = run_dir / "provenance"
    provenance_inputs = {
        name: file_sha256(provenance_dir / name)
        for name in (
            "original_process.stdout.log",
            "original_process.stderr.log",
            "provenance_recovery.json",
            "tokenizer_warning_audit.json",
        )
    }
    payload = {
        "status": "PASS",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "audit_type": "deterministic_engineering_closure",
        "audit_artifact_schema_version": "b500-engineering-audit-v2",
        "audit_code_provenance": audit_code_provenance,
        "source_run_id": run_manifest["run_id"],
        "source_run_git_commit": run_manifest["git_commit"],
        "evaluation_code_git_commit": evaluation_manifest[
            "resume_code_git_commit"
        ],
        "evaluation_directory": args.evaluation_directory_name,
        "row_audit": row_audit,
        "pinned_dataset_rescore": {
            "status": "PASS",
            "dataset_repo_id": gsm_config["repo_id"],
            "dataset_revision": gsm_config["revision"],
            "rescored_row_count": len(rows),
            "all_saved_rows_match_recomputation": True,
        },
        "artifact_hashes": {
            **config_hashes,
            "raw_outputs_sha256": raw_outputs_sha256,
            "adapter_model_sha256": adapter_model_sha256,
            "all_hash_contracts_match": True,
        },
        "serialized_adapter": serialized_adapter,
        "independent_adapter_reload": reload_audit,
        "provenance_inputs": provenance_inputs,
        "claim_boundary": (
            "This audit verifies one random B=500 engineering run only. "
            "It does not compare selectors or estimate seed variance."
        ),
    }
    _write_json_exclusive(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
