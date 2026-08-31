"""Run base16 or archived-adapter128 natural-batch1 v8 canary."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    canonical_jsonl_bytes,
    file_sha256,
    read_json,
    read_jsonl,
    write_exclusive_or_verify,
)
from eg_sft.evaluation.phase2_v8_canary import (  # noqa: E402
    FULL_LEVELS,
    HISTORICAL_ADAPTER_LEVELS,
    HISTORICAL_BASE_LEVELS,
    canonical_first_eos,
    compare_v8_signatures,
    v8_signature,
)
from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract  # noqa: E402
from eg_sft.experiment.phase2_v8_environment import (  # noqa: E402
    validate_v8_environment_manifest,
)
from eg_sft.experiment.phase2_v8_snapshot import (  # noqa: E402
    configure_frozen_snapshot,
    current_single_gpu_identity,
    validate_snapshot_manifest,
)


def _historical_base(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "raw_continuation_ids": row["raw_ids"],
            "first_eos_continuation_ids": row["first_eos_ids"],
            "decoded_canonical_text": row["decoded_text"],
        }
        for row in rows
    ]


def _historical_adapter(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "decoded_canonical_text": row["decoded_text"],
            "parser_input_text": row.get("parser_input_text", row["decoded_text"]),
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_clean_common24_v8_canonical.json")
    )
    parser.add_argument(
        "--canary-contract", type=Path, default=Path("configs/phase2_v8_canary_contract.json")
    )
    parser.add_argument(
        "--role",
        choices=("base16", "adapter128", "training_anchor128"),
        required=True,
    )
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-files-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-files-manifest", type=Path, required=True)
    parser.add_argument("--worker-id", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--new-block-token-anchor", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    canary_contract_path = args.canary_contract.resolve()
    contract = read_json(canary_contract_path)
    expected_contract_sha = config["runtime_contracts"]["canary"]["sha256"]
    if file_sha256(canary_contract_path) != expected_contract_sha:
        raise ValueError("v8 canary contract SHA changed")
    environment = read_json(args.environment_manifest.resolve())
    environment_sha = validate_v8_environment_manifest(environment)
    if environment.get("worker_id") != args.worker_id:
        raise ValueError("v8 canary worker/environment identity mismatch")
    if environment.get("gpu", {}).get("uuid") != args.expected_gpu_uuid:
        raise ValueError("v8 canary expected GPU UUID changed")
    if args.role == "base16":
        spec = contract["base_exact_canary"]
        reference_path = (ROOT / spec["reference_path"]).resolve()
        manifest_path = (ROOT / spec["manifest_path"]).resolve()
        historical_reference = _historical_base(read_jsonl(reference_path))
        historical_levels = HISTORICAL_BASE_LEVELS
        expected_count = 16
    else:
        spec = contract["archived_adapter_semantic_bridge"]
        reference_path = (ROOT / spec["reference_path"]).resolve()
        manifest_path = (ROOT / spec["manifest_path"]).resolve()
        historical_reference = _historical_adapter(read_jsonl(reference_path))
        historical_levels = HISTORICAL_ADAPTER_LEVELS
        expected_count = 128
        if args.adapter_dir is None:
            raise ValueError("adapter canary requires --adapter-dir")
        if args.role == "adapter128":
            adapter_model = args.adapter_dir.resolve() / "adapter_model.safetensors"
            if file_sha256(adapter_model) != spec["adapter_model_sha256"]:
                raise ValueError("v8 archived adapter SHA changed")
    if file_sha256(reference_path) != spec["reference_sha256"]:
        raise ValueError("v8 canary reference SHA changed")
    if file_sha256(manifest_path) != spec["manifest_sha256"]:
        raise ValueError("v8 canary reference manifest SHA changed")
    reference_manifest = read_json(manifest_path)
    reference_ids = [str(row["record_id"]) for row in historical_reference]
    if reference_manifest.get("record_ids") != reference_ids:
        raise ValueError("v8 canary reference record IDs changed")
    if len(historical_reference) != expected_count:
        raise ValueError("v8 historical reference count changed")
    snapshot = configure_frozen_snapshot(args.model_snapshot)
    model_manifest_path = args.model_files_manifest.resolve()
    tokenizer_manifest_path = args.tokenizer_files_manifest.resolve()
    validate_snapshot_manifest(
        snapshot=snapshot, manifest=read_json(model_manifest_path)
    )
    validate_snapshot_manifest(
        snapshot=snapshot, manifest=read_json(tokenizer_manifest_path)
    )
    if environment["model"]["files_manifest_sha256"] != file_sha256(
        model_manifest_path
    ):
        raise ValueError("v8 canary model tree/environment mismatch")
    if environment["tokenizer"]["files_manifest_sha256"] != file_sha256(
        tokenizer_manifest_path
    ):
        raise ValueError("v8 canary tokenizer tree/environment mismatch")
    if args.contract_only:
        print(json.dumps({"status": "READY", "role": args.role, "record_count": expected_count, "environment_contract_sha256": environment_sha, "gpu_accessed": False}, sort_keys=True))
        return
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("fresh v8 canary output already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from eg_sft.data.public_gsm8k import validate_gsm8k_source_row
    from eg_sft.evaluation.gsm8k_generation import build_evaluation_prompt, score_generation

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("v8 canary requires one BF16 CUDA GPU")
    current_gpu = current_single_gpu_identity()
    if current_gpu != environment["gpu"]:
        raise ValueError("v8 canary current GPU/environment mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    set_seed(20260722)
    anchor = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=args.config.resolve(),
        cell_id="v8_rep1_random_common_mix_train17",
    )
    frozen = {row["record_id"]: row for row in read_jsonl(anchor["data_dir"] / "gsm8k_records.jsonl")}
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
    base = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model = (
        PeftModel.from_pretrained(base, args.adapter_dir.resolve(), is_trainable=False)
        if args.role in {"adapter128", "training_anchor128"}
        else base
    )
    model = model.to("cuda").eval()
    model.config.use_cache = True
    signatures = []
    raw_rows = []
    started = time.perf_counter()
    for reference in historical_reference:
        record = frozen[str(reference["record_id"])]
        source_row = dict(source[int(record["source_index"])])
        validate_gsm8k_source_row(record, source_row)
        prompt = build_evaluation_prompt(source_row["question"])
        encoded = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=512).to("cuda")
        prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
        attention = [int(value) for value in encoded["attention_mask"][0].tolist()]
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
        raw_ids = [int(value) for value in generated[0, len(prompt_ids):].tolist()]
        first_eos = canonical_first_eos(raw_ids, [int(tokenizer.eos_token_id)])
        text = tokenizer.decode(first_eos, skip_special_tokens=True).strip()
        row = score_generation(record=record, gold_answer_text=source_row["answer"], generated_text=text)
        row.update({"raw_continuation_ids": raw_ids, "first_eos_continuation_ids": first_eos, "parser_input_text": text})
        raw_rows.append(row)
        signatures.append(v8_signature(row, prompt_ids=prompt_ids, attention_mask=attention))
    historical = compare_v8_signatures(reference=historical_reference, candidate=signatures, levels=historical_levels, expected_count=expected_count)
    token_anchor_report = None
    if args.new_block_token_anchor is not None:
        token_anchor_report = compare_v8_signatures(reference=read_jsonl(args.new_block_token_anchor.resolve()), candidate=signatures, levels=FULL_LEVELS, expected_count=expected_count)
    raw_path = output_dir / "raw_outputs.jsonl"
    signatures_path = output_dir / "signatures.jsonl"
    write_exclusive_or_verify(raw_path, canonical_jsonl_bytes(raw_rows))
    write_exclusive_or_verify(signatures_path, canonical_jsonl_bytes(signatures))
    if args.new_block_token_anchor is None:
        write_exclusive_or_verify(output_dir / "new_block_token_anchor.jsonl", canonical_jsonl_bytes(signatures))
    passed = historical["status"] == "PASS" and (token_anchor_report is None or token_anchor_report["status"] == "PASS")
    audit = {
        "schema_version": "phase2-v8-canary-audit-v1",
        "status": "PASS" if passed else "FAIL",
        "role": args.role,
        "record_count": expected_count,
        "historical_bridge": historical,
        "new_block_token_anchor": token_anchor_report,
        "environment_contract_sha256": environment_sha,
        "environment_manifest_sha256": file_sha256(args.environment_manifest.resolve()),
        "raw_outputs_sha256": file_sha256(raw_path),
        "signatures_sha256": file_sha256(signatures_path),
        "generation_seconds": time.perf_counter() - started,
        "historical_token_exact_claimed": False,
        "gpu_accessed": True,
        "accuracy_withheld": True,
    }
    write_exclusive_or_verify(output_dir / "canary_audit.json", canonical_json_bytes(audit))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError(f"v8 {args.role} canary failed")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
