"""Run the frozen, accuracy-blind 16-item batch-shape falsification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.batch_shape_falsification16 import (  # noqa: E402
    PASS_IDS,
    audit_phase,
    canonical_json_bytes,
    canonical_token_ids,
    derive_selection,
    effective_token_count,
    file_sha256,
    first_eos_index,
    read_json,
    validate_config,
    validate_selection_against_config,
    validate_source_runs,
    write_exclusive_or_verify,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/batch_shape_falsification16_v1.json"),
    )
    parser.add_argument("--source-smoke-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pass-id", choices=PASS_IDS)
    parser.add_argument("--audit-phase", choices=("baseline_repeat", "bf16_repeat", "width_effect", "final_mechanism"))
    parser.add_argument("--fp32-preflight", action="store_true")
    return parser.parse_args()


def _sha256_values(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _pass_spec(config: dict[str, Any], pass_id: str) -> dict[str, Any]:
    matches = [row for row in config["passes"] if row["pass_id"] == pass_id]
    if len(matches) != 1:
        raise ValueError(f"pass binding missing or duplicated: {pass_id}")
    return dict(matches[0])


def _frozen_pretrained_source(
    *, config: dict[str, Any], section: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Resolve the frozen Hub revision without leaking a machine-local path."""

    binding = config[section]
    repo_id = str(binding["repo_id"])
    revision = str(binding["revision"])
    snapshot_value = os.environ.get("EG_SFT_OFFLINE_MODEL_SNAPSHOT")
    if not snapshot_value:
        return (
            repo_id,
            {"revision": revision},
            {
                "source_type": "hub_repo_revision",
                "repo_id": repo_id,
                "revision": revision,
            },
        )

    snapshot = Path(snapshot_value).expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError("EG_SFT_OFFLINE_MODEL_SNAPSHOT is not a directory")
    if snapshot.name != revision:
        raise ValueError(
            "offline snapshot directory does not match the frozen revision"
        )
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise ValueError(f"offline snapshot is incomplete: {missing}")
    return (
        str(snapshot),
        {"local_files_only": True},
        {
            "source_type": "frozen_local_snapshot",
            "repo_id": repo_id,
            "revision": revision,
            "required_file_sha256": {
                name: file_sha256(snapshot / name) for name in required
            },
        },
    )


def _load_protocol_artifacts(output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = read_json(output_root / "selection_manifest.json")
    tokenization = read_json(output_root / "tokenization_manifest.json")
    if selection.get("selected_count") != 16:
        raise ValueError("prepared selection is not the frozen 16-item set")
    if int(tokenization.get("fixed_padding_width", 0)) <= 0:
        raise ValueError("fixed padding width is invalid")
    return selection, tokenization


def _build_entries_for_selection(
    *,
    config: dict[str, Any],
    selection: dict[str, Any],
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the frozen 16 GSM8K inputs directly from source evidence.

    This intentionally avoids any Phase-1 matrix dependency: the diagnostic
    is bound to the already-audited base/batch1 manifest and its pinned GSM8K
    revision, then verifies each selected record against the public source.
    """

    from datasets import load_dataset

    from eg_sft.data.public_gsm8k import sha256_text, validate_gsm8k_source_row
    from eg_sft.evaluation.gsm8k_generation import (
        PROMPT_VERSION,
        build_evaluation_prompt,
    )

    source_rows = validate_source_runs(source_root=source_root, config=config)
    reference_by_id = {
        str(row["record_id"]): row for row in source_rows[1]
    }
    manifest_path = source_root / "smoke128__base__b1" / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("base_model_repo_id") != config["model"]["repo_id"]:
        raise ValueError("source base model repo changed")
    if manifest.get("base_model_revision") != config["model"]["revision"]:
        raise ValueError("source base model revision changed")
    dataset_revision = str(
        manifest.get("dataset_revisions", {}).get("openai/gsm8k", "")
    )
    if len(dataset_revision) != 40:
        raise ValueError("source GSM8K revision is missing or invalid")
    gsm_source = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision=dataset_revision,
    )
    selected_ids = list(selection["selected_record_ids"])
    if any(record_id not in reference_by_id for record_id in selected_ids):
        raise ValueError("selected record is absent from the frozen 128 entries")
    entries: list[dict[str, Any]] = []
    for record_id in selected_ids:
        reference = reference_by_id[record_id]
        source_index = int(reference["source_index"])
        source_row = dict(gsm_source[source_index])
        record = {
            "record_id": record_id,
            "source_split": "test",
            "source_index": source_index,
            "protocol_split": "held_out_test",
            "question_sha256": str(reference["question_sha256"]),
            "answer_sha256": sha256_text(source_row["answer"]),
        }
        validate_gsm8k_source_row(record, source_row)
        question_sha = str(record["question_sha256"])
        expected_id = f"gsm8k-test-{source_index:04d}-{question_sha[:12]}"
        if expected_id != record_id:
            raise ValueError("selected record ID does not match the pinned source")
        if reference.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("source prompt version changed")
        entries.append(
            {
                "dataset": "gsm8k",
                "record": record,
                "prompt": build_evaluation_prompt(source_row["question"]),
                "gold": source_row["answer"],
            }
        )
    return entries, {
        "dataset_repo_id": "openai/gsm8k",
        "dataset_revision": dataset_revision,
        "source_manifest_sha256": file_sha256(manifest_path),
        "prompt_version": PROMPT_VERSION,
    }


def _prepare(
    *,
    config_path: Path,
    config: dict[str, Any],
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_rows = validate_source_runs(source_root=source_root, config=config)
    eos = int(config["tokenizer"]["eos_token_id"])
    selection = derive_selection(source_rows=source_rows, eos_token_id=eos)
    validate_selection_against_config(selection=selection, config=config)
    entries, input_runtime = _build_entries_for_selection(
        config=config,
        selection=selection,
        source_root=source_root,
    )

    from transformers import AutoTokenizer

    tokenizer_source, tokenizer_source_kwargs, tokenizer_source_manifest = (
        _frozen_pretrained_source(config=config, section="tokenizer")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        **tokenizer_source_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = str(config["tokenizer"]["padding_side"])
    if tokenizer.eos_token_id != eos or tokenizer.pad_token_id != eos:
        raise ValueError("runtime tokenizer EOS/pad binding changed")
    max_input_length = int(config["generation"]["max_input_length"])
    prompts = [str(entry["prompt"]) for entry in entries]
    encoded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_input_length,
    )
    lengths = [len(row) for row in encoded["input_ids"]]
    fixed_width = max(lengths)
    if fixed_width <= 0 or fixed_width > max_input_length:
        raise ValueError("derived fixed padding width is outside the frozen limit")
    tokenization = {
        "schema_version": "batch-shape-falsification16-tokenization-v1",
        "config_sha256": file_sha256(config_path),
        "tokenizer_repo_id": config["tokenizer"]["repo_id"],
        "tokenizer_revision": config["tokenizer"]["revision"],
        "tokenizer_runtime_source": tokenizer_source_manifest,
        "padding_side": tokenizer.padding_side,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "selected_record_ids": selection["selected_record_ids"],
        "unpadded_input_lengths": lengths,
        "fixed_padding_width": fixed_width,
        "position_ids_policy": config["generation"]["position_ids_policy"],
        "explicit_position_ids_passed": False,
        "input_runtime": input_runtime,
    }
    protocol = {
        "schema_version": config["schema_version"],
        "config_sha256": file_sha256(config_path),
        "source_root_name": source_root.name,
        "source_bindings": config["source_smoke"]["required_runs"],
        "selected_record_ids": selection["selected_record_ids"],
        "passes": config["passes"],
        "gates": config["gates"],
        "forbidden": config["forbidden"],
        "accuracy_withheld": True,
        "batch_gt_1_authorized": False,
        "gpu_accessed": False,
    }
    write_exclusive_or_verify(output_root / "selection_manifest.json", canonical_json_bytes(selection))
    write_exclusive_or_verify(output_root / "tokenization_manifest.json", canonical_json_bytes(tokenization))
    write_exclusive_or_verify(output_root / "protocol_manifest.json", canonical_json_bytes(protocol))
    return {
        "status": "PREPARED",
        "selected_count": 16,
        "fixed_padding_width": fixed_width,
        "gpu_accessed": False,
    }


def _runtime_determinism(torch: Any) -> dict[str, Any]:
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "UNSET"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


def _validate_determinism(config: dict[str, Any], runtime: dict[str, Any]) -> None:
    expected = config["determinism"]
    for field in (
        "cublas_workspace_config",
        "torch_deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
    ):
        if runtime[field] != expected[field]:
            raise ValueError(f"determinism environment changed: {field}")


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    padding_policy: str,
    fixed_width: int,
    config: dict[str, Any],
    device: Any,
) -> tuple[list[str], list[list[int]], dict[str, list[list[int]]]]:
    if padding_policy == "longest_per_batch":
        padding: bool | str = True
        max_length = int(config["generation"]["max_input_length"])
    elif padding_policy == "fixed_selection_max":
        padding = "max_length"
        max_length = fixed_width
    else:
        raise ValueError(f"unknown padding policy: {padding_policy}")
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=padding,
        truncation=True,
        max_length=max_length,
    ).to(device)
    input_width = int(encoded["input_ids"].shape[1])
    with __import__("torch").inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    raw_rows = [
        [int(value) for value in row[input_width:].tolist()]
        for row in generated
    ]
    texts = [
        tokenizer.decode(
            canonical_token_ids(row, int(config["tokenizer"]["eos_token_id"])),
            skip_special_tokens=True,
        ).strip()
        for row in raw_rows
    ]
    inputs = [[int(value) for value in row] for row in encoded["input_ids"].tolist()]
    masks = [[int(value) for value in row] for row in encoded["attention_mask"].tolist()]
    positions = []
    for mask in masks:
        running = -1
        row_positions = []
        for keep in mask:
            if keep:
                running += 1
                row_positions.append(running)
            else:
                row_positions.append(0)
        positions.append(row_positions)
    return texts, raw_rows, {
        "input_ids": inputs,
        "attention_mask": masks,
        "derived_position_ids": positions,
    }


def _run_pass(
    *,
    config_path: Path,
    config: dict[str, Any],
    source_root: Path,
    output_root: Path,
    pass_id: str,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / pass_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = read_json(metrics_path)
        raw_path = run_dir / "raw_outputs.jsonl"
        if metrics.get("raw_outputs_sha256") != file_sha256(raw_path):
            raise ValueError("completed pass raw output changed")
        return {"status": "REPLAY_VERIFIED", "pass_id": pass_id, "gpu_accessed": False}
    selection, tokenization = _load_protocol_artifacts(output_root)
    entries, _ = _build_entries_for_selection(
        config=config,
        selection=selection,
        source_root=source_root,
    )
    spec = _pass_spec(config, pass_id)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("16-item falsification requires one CUDA GPU")
    if spec["dtype"] == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unavailable")
        dtype = torch.bfloat16
    elif spec["dtype"] == "float32":
        dtype = torch.float32
    else:
        raise ValueError("unsupported pass dtype")
    runtime_determinism = _runtime_determinism(torch)
    _validate_determinism(config, runtime_determinism)
    set_seed(int(config["determinism"]["seed"]))
    device = torch.device("cuda:0")
    tokenizer_source, tokenizer_source_kwargs, tokenizer_source_manifest = (
        _frozen_pretrained_source(config=config, section="tokenizer")
    )
    model_source, model_source_kwargs, model_source_manifest = (
        _frozen_pretrained_source(config=config, section="model")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        **tokenizer_source_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = str(config["tokenizer"]["padding_side"])
    if tokenizer.pad_token_id != int(config["tokenizer"]["pad_token_id"]):
        raise ValueError("runtime pad token changed")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=config["model"]["attn_implementation"],
        **model_source_kwargs,
    ).to(device)
    model.eval()
    model.config.use_cache = True
    from eg_sft.evaluation.gsm8k_generation import score_generation

    rows: list[dict[str, Any]] = []
    generation_seconds = 0.0
    batch_size = int(spec["batch_size"])
    fixed_width = int(tokenization["fixed_padding_width"])
    eos = int(config["tokenizer"]["eos_token_id"])
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        prompts = [str(entry["prompt"]) for entry in batch]
        generation_started = time.perf_counter()
        texts, raw_token_rows, encoded = _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            padding_policy=str(spec["padding_policy"]),
            fixed_width=fixed_width,
            config=config,
            device=device,
        )
        generation_seconds += time.perf_counter() - generation_started
        for offset, (entry, text, raw_ids) in enumerate(
            zip(batch, texts, raw_token_rows, strict=True)
        ):
            batch_position = offset
            input_ids = encoded["input_ids"][offset]
            attention_mask = encoded["attention_mask"][offset]
            derived_positions = encoded["derived_position_ids"][offset]
            row = score_generation(
                record=entry["record"],
                gold_answer_text=entry["gold"],
                generated_text=text,
            )
            row["dataset"] = "gsm8k"
            row.update(
                {
                    "pass_id": pass_id,
                    "global_selected_index": start + offset,
                    "batch_index": start // batch_size,
                    "batch_position": batch_position,
                    "dtype": spec["dtype"],
                    "physical_batch_size": batch_size,
                    "padding_policy": spec["padding_policy"],
                    "padding_side": tokenizer.padding_side,
                    "input_width": len(input_ids),
                    "padding_length": len(input_ids) - sum(attention_mask),
                    "input_ids_sha256": _sha256_values(input_ids),
                    "attention_mask_sha256": _sha256_values(attention_mask),
                    "derived_position_ids_sha256": _sha256_values(derived_positions),
                    "position_ids_policy": config["generation"]["position_ids_policy"],
                    "explicit_position_ids_passed": False,
                    "raw_generated_tensor_ids": raw_ids,
                    "canonical_generated_ids": canonical_token_ids(raw_ids, eos),
                    "first_eos_index": first_eos_index(raw_ids, eos),
                    "raw_generated_token_count": len(raw_ids),
                    "effective_generated_token_count": effective_token_count(raw_ids, eos),
                }
            )
            rows.append(row)
    peak_memory = int(torch.cuda.max_memory_allocated())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    raw_path = run_dir / "raw_outputs.jsonl"
    raw_bytes = _jsonl_bytes(rows)
    write_exclusive_or_verify(raw_path, raw_bytes)
    effective_tokens = sum(int(row["effective_generated_token_count"]) for row in rows)
    metrics = {
        "schema_version": "batch-shape-falsification16-pass-v1",
        "status": "PASS",
        "pass_id": pass_id,
        "config_sha256": file_sha256(config_path),
        "selection_manifest_sha256": file_sha256(output_root / "selection_manifest.json"),
        "tokenization_manifest_sha256": file_sha256(output_root / "tokenization_manifest.json"),
        "record_count": len(rows),
        "dtype": spec["dtype"],
        "physical_batch_size": batch_size,
        "padding_policy": spec["padding_policy"],
        "generation_seconds": generation_seconds,
        "full_wall_seconds": elapsed,
        "examples_per_second": len(rows) / generation_seconds,
        "effective_generated_tokens": effective_tokens,
        "effective_generated_tokens_per_second": effective_tokens / generation_seconds,
        "peak_memory_bytes": peak_memory,
        "raw_outputs_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "model_repo_id": config["model"]["repo_id"],
        "model_revision": config["model"]["revision"],
        "model_runtime_source": model_source_manifest,
        "tokenizer_repo_id": config["tokenizer"]["repo_id"],
        "tokenizer_revision": config["tokenizer"]["revision"],
        "tokenizer_runtime_source": tokenizer_source_manifest,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": str(
            getattr(torch.cuda.get_device_properties(0), "uuid", "UNAVAILABLE")
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "determinism": runtime_determinism,
        "accuracy_withheld": True,
        "accuracy_aggregated": False,
        "batch_gt_1_authorized": False,
    }
    write_exclusive_or_verify(metrics_path, canonical_json_bytes(metrics))
    return {
        "status": "COMPLETE",
        "pass_id": pass_id,
        "record_count": len(rows),
        "gpu_accessed": True,
    }


def _fp32_preflight(output_root: Path) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FP32 preflight requires CUDA")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    required_free_bytes = 12 * 2**30
    report = {
        "schema_version": "batch-shape-falsification16-fp32-preflight-v1",
        "status": "PASS" if free_bytes >= required_free_bytes else "FAIL",
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "required_free_bytes": required_free_bytes,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": str(
            getattr(torch.cuda.get_device_properties(0), "uuid", "UNAVAILABLE")
        ),
        "gpu_accessed": True,
    }
    write_exclusive_or_verify(
        output_root / "audit" / "fp32_preflight.json", canonical_json_bytes(report)
    )
    if report["status"] != "PASS":
        raise RuntimeError("FP32 memory preflight failed")
    return report


def main() -> None:
    args = _arguments()
    modes = (
        args.contract_only,
        args.prepare,
        args.pass_id is not None,
        args.audit_phase is not None,
        args.fp32_preflight,
    )
    if sum(bool(value) for value in modes) != 1:
        raise ValueError("select exactly one execution mode")
    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config)
    source_root = args.source_smoke_root.resolve()
    output_root = args.output_root.resolve()
    if args.contract_only:
        source_rows = validate_source_runs(source_root=source_root, config=config)
        selection = derive_selection(
            source_rows=source_rows,
            eos_token_id=int(config["tokenizer"]["eos_token_id"]),
        )
        validate_selection_against_config(selection=selection, config=config)
        print(
            json.dumps(
                {
                    "status": "READY",
                    "selected_count": 16,
                    "pass_count": 6,
                    "config_sha256": file_sha256(config_path),
                    "gpu_accessed": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.prepare:
        result = _prepare(
            config_path=config_path,
            config=config,
            source_root=source_root,
            output_root=output_root,
        )
    elif args.pass_id is not None:
        try:
            result = _run_pass(
                config_path=config_path,
                config=config,
                source_root=source_root,
                output_root=output_root,
                pass_id=args.pass_id,
            )
        except BaseException as error:
            failure = {
                "schema_version": "batch-shape-falsification16-failure-v1",
                "pass_id": args.pass_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "batch_gt_1_authorized": False,
            }
            write_exclusive_or_verify(
                output_root / "failures" / f"{args.pass_id}.json",
                canonical_json_bytes(failure),
            )
            raise
    elif args.fp32_preflight:
        result = _fp32_preflight(output_root)
    else:
        source_rows = validate_source_runs(source_root=source_root, config=config)
        selection, _ = _load_protocol_artifacts(output_root)
        result = audit_phase(
            phase=args.audit_phase,
            output_root=output_root,
            source_rows=source_rows,
            selection=selection,
            config=config,
        )
        write_exclusive_or_verify(
            output_root / "audit" / f"{args.audit_phase}.json",
            canonical_json_bytes(result),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
