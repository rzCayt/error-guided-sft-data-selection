"""Fresh-process natural-batch1 canary for base or archived parent adapter."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    CANARY_LEVELS,
    SEMANTIC_CANARY_LEVELS,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canary_signature,
    compare_canary_signatures,
    file_sha256,
    read_json,
    read_jsonl,
    validate_reference_manifest,
    write_exclusive_or_verify,
)
from eg_sft.experiment.budget_equivalent_matrix import (  # noqa: E402
    resolve_phase1_contract,
)
from eg_sft.experiment.phase2_v7_environment import (  # noqa: E402
    validate_environment_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_crossed_48cell_v7.json"),
    )
    parser.add_argument(
        "--backend-contract",
        type=Path,
        default=Path("configs/phase2_v7_legacy_batch1_contract.json"),
    )
    parser.add_argument("--role", choices=("base_model_16", "archived_adapter_16"), required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--adapter-token-anchor", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    backend = read_json(args.backend_contract.resolve())
    environment = read_json(args.environment_manifest.resolve())
    environment_contract_sha = validate_environment_manifest(environment)
    role_contract = backend[args.role]
    if args.role == "base_model_16":
        reference_path = (ROOT / role_contract["reference_path"]).resolve()
        manifest_path = (ROOT / role_contract["manifest_path"]).resolve()
        levels = CANARY_LEVELS
    else:
        reference_path = (
            args.adapter_token_anchor.resolve()
            if args.adapter_token_anchor is not None
            else (ROOT / role_contract["historical_semantic_reference_path"]).resolve()
        )
        manifest_path = (
            None
            if args.adapter_token_anchor is not None
            else (ROOT / role_contract["historical_manifest_path"]).resolve()
        )
        levels = (
            CANARY_LEVELS
            if args.adapter_token_anchor is not None
            else SEMANTIC_CANARY_LEVELS
        )
    if manifest_path is not None:
        manifest = read_json(manifest_path)
        if args.role == "base_model_16":
            validate_reference_manifest(manifest=manifest, reference_path=reference_path)
    reference = read_jsonl(reference_path)
    if len(reference) != 16:
        raise ValueError("canary reference must contain exactly 16 rows")
    if args.role == "archived_adapter_16":
        if args.adapter_dir is None:
            raise ValueError("archived adapter canary requires --adapter-dir")
        adapter_model = args.adapter_dir.resolve() / "adapter_model.safetensors"
        if file_sha256(adapter_model) != role_contract["adapter_model_sha256"]:
            raise ValueError("archived adapter SHA-256 changed")
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "role": args.role,
                    "comparison_levels": list(levels),
                    "reference_sha256": file_sha256(reference_path),
                    "environment_contract_sha256": environment_contract_sha,
                    "gpu_accessed": False,
                },
                sort_keys=True,
            )
        )
        return
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("fresh canary output directory already exists")
    output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from eg_sft.data.public_gsm8k import validate_gsm8k_source_row
    from eg_sft.evaluation.gsm8k_generation import build_evaluation_prompt, score_generation

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase-2 canary requires one BF16 CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(20260722)
    snapshot = args.model_snapshot.resolve(strict=True)
    if snapshot.name != "8faed761d45a263340a0528343f099c05c9a4323":
        raise ValueError("canary model snapshot revision changed")
    anchor = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=args.config.resolve(),
        cell_id="rep1_random_common_mix_train29",
    )
    frozen_records = {
        row["record_id"]: row
        for row in read_jsonl(anchor["data_dir"] / "gsm8k_records.jsonl")
    }
    source = load_dataset(
        anchor["protocol"]["datasets"]["gsm8k"]["repo_id"],
        anchor["protocol"]["datasets"]["gsm8k"]["config"],
        split="test",
        revision=anchor["protocol"]["datasets"]["gsm8k"]["revision"],
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = (
        PeftModel.from_pretrained(base, args.adapter_dir.resolve(), is_trainable=False)
        if args.role == "archived_adapter_16"
        else base
    ).to("cuda")
    model.eval()
    model.config.use_cache = True
    rows = []
    started = time.perf_counter()
    for expected in reference:
        record_id = str(expected["record_id"])
        record = frozen_records[record_id]
        source_row = dict(source[int(record["source_index"])])
        validate_gsm8k_source_row(record, source_row)
        prompt = build_evaluation_prompt(source_row["question"])
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        ).to("cuda")
        input_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        raw_ids = [int(value) for value in generated[0, input_width:].tolist()]
        try:
            eos_end = raw_ids.index(int(tokenizer.eos_token_id)) + 1
        except ValueError:
            eos_end = len(raw_ids)
        canonical_ids = raw_ids[:eos_end]
        text = tokenizer.decode(canonical_ids, skip_special_tokens=True).strip()
        row = score_generation(
            record=record,
            gold_answer_text=source_row["answer"],
            generated_text=text,
        )
        row.update(
            {
                "dataset": "gsm8k",
                "raw_generated_tensor_ids": raw_ids,
                "canonical_generated_ids": canonical_ids,
                "canonical_decoded_text": text,
                "parser_input": text,
            }
        )
        rows.append(row)
    signatures = [
        canary_signature(row, eos_token_id=int(tokenizer.eos_token_id)) for row in rows
    ]
    comparison = compare_canary_signatures(
        reference=reference,
        candidate=signatures,
        comparison_levels=levels,
    )
    audit = {
        "schema_version": "phase2-v7-canary-audit-v1",
        "status": comparison["status"],
        "role": args.role,
        "record_count": 16,
        "exact_all_levels": comparison["exact_all_levels"],
        "comparison_levels": list(levels),
        "comparison": comparison,
        "environment_contract_sha256": environment_contract_sha,
        "environment_manifest_sha256": file_sha256(args.environment_manifest.resolve()),
        "raw_outputs_sha256": "PENDING",
        "signatures_sha256": "PENDING",
        "generation_seconds": time.perf_counter() - started,
        "gpu_accessed": True,
        "accuracy_withheld": True,
    }
    raw_path = output_dir / "raw_outputs.jsonl"
    signature_path = output_dir / "canary_signatures.jsonl"
    write_exclusive_or_verify(raw_path, canonical_jsonl_bytes(rows))
    write_exclusive_or_verify(signature_path, canonical_jsonl_bytes(signatures))
    audit["raw_outputs_sha256"] = file_sha256(raw_path)
    audit["signatures_sha256"] = file_sha256(signature_path)
    if args.role == "archived_adapter_16" and args.adapter_token_anchor is None:
        anchor_path = output_dir / "adapter_token_anchor.jsonl"
        write_exclusive_or_verify(anchor_path, canonical_jsonl_bytes(signatures))
        audit["adapter_token_anchor_sha256"] = file_sha256(anchor_path)
        audit["anchor_bootstrapped_after_historical_semantic_pass"] = (
            comparison["status"] == "PASS"
        )
    write_exclusive_or_verify(
        output_dir / "canary_audit.json", canonical_json_bytes(audit)
    )
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    if audit["status"] != "PASS":
        raise RuntimeError(f"{args.role} canary failed")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
