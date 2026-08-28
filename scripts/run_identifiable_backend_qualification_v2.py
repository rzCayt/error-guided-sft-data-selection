"""Run or adjudicate batched Transformers qualification v2.

``--contract-only``, ``--select-smoke`` and ``--audit-final`` are CPU-only.
CUDA/PyTorch/Transformers are imported lazily only for an actual generation
run.  Every generation invocation targets one deterministic run directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.evaluation.identifiable_backend_qualification_v2 import (  # noqa: E402
    BATCH_SIZES,
    MODEL_IDS,
    adapter_binding,
    audit_final,
    audit_smoke128,
    canonical_json_bytes,
    file_sha256,
    finalize_run,
    read_json,
    read_jsonl,
    record_attempt_finish,
    record_attempt_start,
    run_dir,
    select_smoke_best,
    validate_adapter_root,
    validate_config,
    validate_prefix,
    validate_stop_after_records_request,
    write_exclusive_or_verify,
    write_failure_artifact,
    write_replay_probe,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/identifiable_backend_qualification_v2.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/identifiable-backend-qualification-v2"),
    )
    parser.add_argument("--stage", choices=("smoke128", "confirm512", "shadow3841"))
    parser.add_argument("--model-id", choices=MODEL_IDS)
    parser.add_argument("--batch-size", type=int, choices=BATCH_SIZES)
    parser.add_argument(
        "--adapter-root",
        action="append",
        default=[],
        metavar="ADAPTER_ID=EXTRACTED_RUN_DIR",
    )
    parser.add_argument("--shadow-adapter-id", choices=MODEL_IDS[1:])
    parser.add_argument("--stop-after-records", type=int)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--select-smoke", action="store_true")
    parser.add_argument("--audit-smoke128", action="store_true")
    parser.add_argument("--audit-final", action="store_true")
    return parser.parse_args()


def _adapter_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        adapter_id, separator, raw_path = value.partition("=")
        if not separator or adapter_id not in MODEL_IDS[1:] or not raw_path:
            raise ValueError("--adapter-root must be AUDITED_ADAPTER_ID=RUN_DIR")
        if adapter_id in result:
            raise ValueError(f"adapter root repeated: {adapter_id}")
        result[adapter_id] = Path(raw_path).resolve()
    return result


def _allowed_batch(
    *, stage: str, batch_size: int, output_root: Path
) -> None:
    if stage == "smoke128":
        return
    selection = read_json(output_root / "smoke_selection.json")
    if selection.get("status") != "PASS":
        raise ValueError("smoke selection has not passed")
    if batch_size not in {1, int(selection["best_batch_size"])}:
        raise ValueError("confirm/shadow permits only batch1 plus smoke best")


def _build_entries(*, matrix_path: Path, record_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse the frozen prompt/score contracts to build one ordered task stream."""

    from datasets import load_dataset

    from eg_sft.data.public_gsm8k import validate_gsm8k_source_row
    from eg_sft.evaluation.arithmetic_ood import build_ood_prompt
    from eg_sft.evaluation.gsm8k_generation import build_evaluation_prompt
    from eg_sft.experiment.budget_equivalent_matrix import resolve_phase1_contract
    from eg_sft.experiment.budget_equivalent_ood_runtime import (
        OOD_DATASETS,
        resolve_ood_contract,
        validate_source_row,
    )
    from eg_sft.training.b500 import read_jsonl as read_training_jsonl

    anchor = resolve_phase1_contract(
        repo_root=ROOT,
        config_path=matrix_path,
        cell_id="rep1_random_common_mix_train29",
    )
    evaluation = anchor["config"]["evaluation"]
    protocol = anchor["protocol"]
    gsm_records = sorted(
        [
            row
            for row in read_training_jsonl(anchor["data_dir"] / "gsm8k_records.jsonl")
            if row["protocol_split"] == evaluation["split"]
        ],
        key=lambda row: (row["source_index"], row["record_id"]),
    )
    gsm_spec = protocol["datasets"]["gsm8k"]
    gsm_source = load_dataset(
        gsm_spec["repo_id"],
        gsm_spec["config"],
        split="test",
        revision=gsm_spec["revision"],
    )
    entries: list[dict[str, Any]] = []
    for record in gsm_records:
        source_row = dict(gsm_source[int(record["source_index"])])
        validate_gsm8k_source_row(record, source_row)
        entries.append(
            {
                "dataset": "gsm8k",
                "record": record,
                "prompt": build_evaluation_prompt(source_row["question"]),
                "gold": source_row["answer"],
            }
        )
    dataset_revisions = {gsm_spec["repo_id"]: gsm_spec["revision"]}
    for dataset in OOD_DATASETS:
        contract = resolve_ood_contract(
            repo_root=ROOT, matrix_config_path=matrix_path, dataset=dataset
        )
        spec = contract["source"]
        source = load_dataset(
            spec["repo_id"],
            spec["config"],
            split=spec["split"],
            revision=spec["revision"],
        )
        dataset_revisions[spec["repo_id"]] = spec["revision"]
        for record in contract["records"]:
            source_row = dict(source[int(record["source_index"])])
            gold = validate_source_row(
                record=record,
                raw_row=source_row,
                answer_field=str(spec["answer_field"]),
            )
            entries.append(
                {
                    "dataset": dataset,
                    "record": record,
                    "prompt": build_ood_prompt(dataset, source_row),
                    "gold": gold,
                }
            )
    if len(entries) != 3841:
        raise ValueError("frozen four-task record count changed")
    return entries[:record_limit], {
        "evaluation": evaluation,
        "dataset_revisions": dataset_revisions,
    }


def _score(entry: dict[str, Any], text: str) -> dict[str, Any]:
    if entry["dataset"] == "gsm8k":
        from eg_sft.evaluation.gsm8k_generation import score_generation

        row = score_generation(
            record=entry["record"],
            gold_answer_text=entry["gold"],
            generated_text=text,
        )
    else:
        from eg_sft.evaluation.arithmetic_ood import score_ood_generation

        row = score_ood_generation(
            record=entry["record"],
            gold_value=entry["gold"],
            generated_text=text,
        )
    row["dataset"] = entry["dataset"]
    return row


def _actual_run(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_path: Path,
    adapter_roots: dict[str, Path],
) -> dict[str, Any]:
    if args.stage is None or args.model_id is None or args.batch_size is None:
        raise ValueError("actual run requires --stage, --model-id and --batch-size")
    if args.stage == "shadow3841" and args.model_id != args.shadow_adapter_id:
        raise ValueError("shadow run model must equal --shadow-adapter-id")
    _allowed_batch(
        stage=args.stage,
        batch_size=args.batch_size,
        output_root=args.output_root.resolve(),
    )
    record_limit = int(config["stages"][args.stage]["record_count"])
    directory = run_dir(
        root=args.output_root,
        stage=args.stage,
        model_id=args.model_id,
        batch_size=args.batch_size,
    )
    directory.mkdir(parents=True, exist_ok=True)

    adapter_hashes: dict[str, str | None] = {
        "formal_audit_sha256": None,
        "training_metrics_sha256": None,
        "adapter_model_sha256": None,
    }
    adapter_model_path: Path | None = None
    if args.model_id != "base":
        if args.model_id not in adapter_roots:
            raise ValueError(f"missing --adapter-root for {args.model_id}")
        binding = adapter_binding(config, args.model_id)
        adapter_hashes = validate_adapter_root(
            adapter_root=adapter_roots[args.model_id], binding=binding
        )
        adapter_model_path = (
            adapter_roots[args.model_id] / "training_complete" / "adapter"
        )

    contract = {
        "schema_version": config["schema_version"],
        "stage": args.stage,
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "record_count": record_limit,
        "base_model_repo_id": config["base_model"]["repo_id"],
        "base_model_revision": config["base_model"]["revision"],
        "tokenizer_repo_id": config["tokenizer"]["repo_id"],
        "tokenizer_revision": config["tokenizer"]["revision"],
        "config_sha256": file_sha256(config_path),
        "matrix_sha256": config["matrix"]["sha256"],
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(
            ROOT
            / "src"
            / "eg_sft"
            / "evaluation"
            / "identifiable_backend_qualification_v2.py"
        ),
        "shadow_adapter_id": args.shadow_adapter_id,
        **adapter_hashes,
        "accuracy_withheld": True,
    }
    write_exclusive_or_verify(
        directory / "contract.json", canonical_json_bytes(contract)
    )

    # A repeated completed call performs a real byte-level non-overwrite probe
    # without importing or accessing CUDA.
    if (directory / "metrics.json").is_file():
        probe = write_replay_probe(directory)
        return {"status": "REPLAY_VERIFIED", "probe": probe.name, "gpu_accessed": False}

    from run_b500_formal_resumable import _global_job_lock

    with _global_job_lock(directory):
        existing = read_jsonl(directory / "raw_outputs.jsonl")
        entries, runtime = _build_entries(
            matrix_path=(ROOT / config["matrix"]["path"]).resolve(),
            record_limit=record_limit,
        )
        frozen_ids = [str(entry["record"]["record_id"]) for entry in entries]
        start_index = validate_prefix(rows=existing, frozen_ids=frozen_ids)
        if args.stop_after_records is not None:
            if not start_index < args.stop_after_records < record_limit:
                raise ValueError(
                    "--stop-after-records must be above the prefix and below completion"
                )
            end_limit = args.stop_after_records
        else:
            end_limit = record_limit

        attempt_id = record_attempt_start(
            run_directory=directory,
            start_index=start_index,
            stop_after_records=args.stop_after_records,
        )
        wall_started = time.perf_counter()
        generation_seconds = 0.0
        generated_tokens = 0
        end_index = start_index
        peak_memory = 0
        try:
            # GPU libraries are intentionally imported only here.
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

            from eg_sft.evaluation.cloud_v2_batching import (
                append_jsonl_rows_fsynced,
                contiguous_record_batches,
            )
            from run_identifiable_base_reference import _batch_outputs

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("qualification requires a BF16 CUDA GPU")
            torch.cuda.reset_peak_memory_stats()
            set_seed(args.seed)
            device = torch.device("cuda")
            tokenizer = AutoTokenizer.from_pretrained(
                config["tokenizer"]["repo_id"],
                revision=config["tokenizer"]["revision"],
                use_fast=True,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(
                config["base_model"]["repo_id"],
                revision=config["base_model"]["revision"],
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            ).to(device)
            if adapter_model_path is not None:
                model = PeftModel.from_pretrained(model, adapter_model_path).to(device)
            model.eval()
            model.config.use_cache = True
            evaluation = runtime["evaluation"]
            for start, batch in contiguous_record_batches(
                records=entries[:end_limit],
                start_index=start_index,
                batch_size=args.batch_size,
            ):
                prompts = [str(entry["prompt"]) for entry in batch]
                generation_started = time.perf_counter()
                texts, token_rows = _batch_outputs(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    batch_size=args.batch_size,
                    max_input_length=int(evaluation["max_input_length"]),
                    max_new_tokens=int(evaluation["max_new_tokens"]),
                    device=device,
                )
                generation_seconds += time.perf_counter() - generation_started
                output_rows = []
                for offset, (entry, text, token_ids) in enumerate(
                    zip(batch, texts, token_rows, strict=True)
                ):
                    row = _score(entry, text)
                    row["global_record_index"] = start + offset
                    row["generated_token_ids"] = [int(value) for value in token_ids]
                    row["generated_token_count"] = len(token_ids)
                    output_rows.append(row)
                    generated_tokens += len(token_ids)
                append_jsonl_rows_fsynced(directory / "raw_outputs.jsonl", output_rows)
                end_index = start + len(batch)
            peak_memory = int(torch.cuda.max_memory_allocated())
            del model
            gc.collect()
            torch.cuda.empty_cache()
            record_attempt_finish(
                run_directory=directory,
                attempt_id=attempt_id,
                start_index=start_index,
                end_index=end_index,
                generation_seconds=generation_seconds,
                full_wall_seconds=time.perf_counter() - wall_started,
                generated_tokens=generated_tokens,
                peak_memory_bytes=peak_memory,
                stopped_early=end_index < record_limit,
                attempt_status="PASS",
            )
        except BaseException:
            current_rows = read_jsonl(directory / "raw_outputs.jsonl")
            end_index = validate_prefix(rows=current_rows, frozen_ids=frozen_ids)
            record_attempt_finish(
                run_directory=directory,
                attempt_id=attempt_id,
                start_index=start_index,
                end_index=end_index,
                generation_seconds=generation_seconds,
                full_wall_seconds=time.perf_counter() - wall_started,
                generated_tokens=generated_tokens,
                peak_memory_bytes=peak_memory,
                stopped_early=True,
                attempt_status="FAIL",
            )
            raise

        if end_index < record_limit:
            return {
                "status": "STOPPED_EARLY",
                "record_count": end_index,
                "metrics_written": False,
                "gpu_accessed": True,
            }
        manifest = {
            **contract,
            "dataset_revisions": runtime["dataset_revisions"],
            "gpu_name": torch.cuda.get_device_name(0),
        }
        metrics = finalize_run(
            run_directory=directory, frozen_ids=frozen_ids, manifest=manifest
        )
        return {
            "status": "COMPLETE",
            "record_count": metrics["record_count"],
            "rows_sha256": metrics["rows_sha256"],
            "gpu_accessed": True,
        }


def main() -> None:
    args = _arguments()
    control_modes = (
        args.contract_only,
        args.select_smoke,
        args.audit_smoke128,
        args.audit_final,
    )
    if sum(bool(value) for value in control_modes) > 1:
        raise ValueError(
            "--contract-only, --select-smoke, --audit-smoke128 and "
            "--audit-final are mutually exclusive"
        )
    config_path = args.config.resolve()
    config = validate_config(repo_root=ROOT, config_path=config_path)
    validate_stop_after_records_request(
        stage=args.stage,
        model_id=args.model_id,
        batch_size=args.batch_size,
        stop_after_records=args.stop_after_records,
        config=config,
    )
    if any(control_modes) and any(
        value is not None
        for value in (args.stage, args.model_id, args.batch_size, args.stop_after_records)
    ):
        raise ValueError("CPU-only control modes cannot include generation-run options")
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "stage": "qualification_v2_contract",
                    "matrix_sha256": config["matrix"]["sha256"],
                    "smoke_models": list(MODEL_IDS),
                    "smoke_batch_sizes": list(BATCH_SIZES),
                    "shadow_adapter_choices": list(MODEL_IDS[1:]),
                    "gpu_accessed": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.select_smoke:
        report = select_smoke_best(root=args.output_root, config=config)
        print(json.dumps(report, sort_keys=True))
        return
    if args.audit_smoke128:
        report = audit_smoke128(root=args.output_root, config=config)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "audit_gpu_accessed": False,
                    "formal_backend_authorized": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.audit_final:
        if args.shadow_adapter_id is None:
            raise ValueError("--audit-final requires --shadow-adapter-id")
        report = audit_final(
            root=args.output_root,
            config=config,
            shadow_adapter_id=args.shadow_adapter_id,
        )
        print(json.dumps({"status": report["status"], "gpu_accessed": False}, sort_keys=True))
        return
    roots = _adapter_roots(args.adapter_root)
    try:
        result = _actual_run(
            args=args,
            config=config,
            config_path=config_path,
            adapter_roots=roots,
        )
    except BaseException as error:
        if args.stage is not None and args.model_id is not None and args.batch_size is not None:
            write_failure_artifact(
                root=args.output_root,
                stage=args.stage,
                model_id=args.model_id,
                batch_size=args.batch_size,
                error=error,
            )
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
