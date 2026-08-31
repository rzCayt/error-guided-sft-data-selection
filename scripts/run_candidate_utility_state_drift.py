#!/usr/bin/env python3
"""Resumable state-conditioned utility probes under the frozen v3 semantics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import (  # noqa: E402
    candidate_prompt_text,
    sha256_text,
    validate_gsm8k_source_row,
)
from eg_sft.experiment.state_drift import (  # noqa: E402
    build_measurement_plan,
    measurement_key,
    validate_resume_rows,
)
from eg_sft.experiment.utility import mean_supervised_token_loss, to_device  # noqa: E402
from eg_sft.training.lora_audit import (  # noqa: E402
    audit_lora_gradients,
    audit_lora_parameters,
)
from eg_sft.training.overfit import (  # noqa: E402
    build_tokenized_overfit_examples,
)
from eg_sft.training.response_only import (  # noqa: E402
    ResponseOnlyCollator,
    tokenize_response_only,
)
from eg_sft.training.tulu import tulu_response_only_parts  # noqa: E402


ZERO_STATE = "zero_initialized_lora"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(row)
    return rows


def write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")



def _validate_candidate(candidate: dict[str, Any], raw_row: dict[str, Any]) -> list[dict[str, str]]:
    messages = raw_row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{candidate['candidate_id']} has invalid messages")
    if sha256_text(candidate_prompt_text(messages)) != candidate["prompt_sha256"]:
        raise ValueError(f"prompt hash mismatch for {candidate['candidate_id']}")
    response = str(messages[-1].get("content", ""))
    if sha256_text(response) != candidate["response_sha256"]:
        raise ValueError(f"response hash mismatch for {candidate['candidate_id']}")
    return messages


def _load_base(repo_id: str, revision: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.to(device)
    return model


def _load_state_model(
    *,
    repo_id: str,
    revision: str,
    state_id: str,
    adapter_dir: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    base = _load_base(repo_id, revision, device)
    if state_id == ZERO_STATE:
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                bias="none",
            ),
        )
    else:
        if adapter_dir is None:
            raise ValueError("trained adapter state requires --adapter-dir")
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    model.to(device)
    return model


def _adapter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in get_peft_model_state_dict(model).items()
    }


def _snapshot_sha256(snapshot: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        tensor = snapshot[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _restore_adapter(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    result = set_peft_model_state_dict(model, snapshot)
    unexpected = list(getattr(result, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"unexpected adapter-state keys: {unexpected}")


def _validate_adapter_dir(
    *, adapter_dir: Path, state_id: str, adapter_index: dict[str, Any]
) -> None:
    matches = [row for row in adapter_index["adapters"] if row["cell_id"] == state_id]
    if len(matches) != 1:
        raise ValueError(f"adapter index has no unique state {state_id}")
    row = matches[0]
    model_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if file_sha256(model_path) != row["adapter_model_sha256"]:
        raise ValueError("adapter model SHA differs from frozen evidence index")
    if file_sha256(config_path) != row["adapter_config_sha256"]:
        raise ValueError("adapter config SHA differs from frozen evidence index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--adapter-index", type=Path, required=True)
    parser.add_argument("--public-config", type=Path, required=True)
    parser.add_argument("--data-manifest-dir", type=Path, required=True)

    parser.add_argument("--state-id", required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--probe-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--zero-adapter-initialization-seed", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cleanup-interval", type=int, default=16)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    preflight_path = args.preflight.resolve()
    panel_path = args.panel.resolve()
    adapter_index_path = args.adapter_index.resolve()
    public_config_path = args.public_config.resolve()
    data_dir = args.data_manifest_dir.resolve()
    protocol = read_json(protocol_path)
    preflight = read_json(preflight_path)
    panel = read_json(panel_path)
    adapter_index = read_json(adapter_index_path)
    if preflight.get("status") != "READY_FOR_GPU_QUALIFICATION":
        raise ValueError("state-dependence v3 preflight is not ready")
    if file_sha256(protocol_path) != preflight["protocol_sha256"]:
        raise ValueError("protocol SHA differs from preflight")
    if file_sha256(panel_path) != preflight["panel_sha256"]:
        raise ValueError("candidate panel SHA differs from preflight")
    if file_sha256(adapter_index_path) != preflight["adapter_index_sha256"]:
        raise ValueError("adapter index SHA differs from preflight")
    allowed_states = {
        ZERO_STATE,
        *protocol["stage_u1_cross_state_transfer"]["initial_adapter_states"],
        *protocol["stage_u1_cross_state_transfer"]["expansion_adapter_states"],
    }
    if args.state_id not in allowed_states:
        raise ValueError(f"state is outside the frozen protocol: {args.state_id}")
    candidate_rows = list(panel["candidates"])
    if args.candidate_limit is not None:
        if args.candidate_limit <= 0 or args.candidate_limit > len(candidate_rows):
            raise ValueError("candidate limit is outside the frozen panel")
        candidate_rows = candidate_rows[: args.candidate_limit]
    candidate_ids = [str(row["candidate_id"]) for row in candidate_rows]
    if args.cleanup_interval <= 0:
        raise ValueError("cleanup interval must be positive")
    if args.state_id == ZERO_STATE:
        expected_init_seed = int(
            protocol["stage_u0a_fixed_state_reliability"][
                "zero_adapter_initialization_seed"
            ]
        )
        if args.zero_adapter_initialization_seed != expected_init_seed:
            raise ValueError("zero-state initialization seed differs from frozen protocol")
    elif args.zero_adapter_initialization_seed is not None:
        raise ValueError("trained adapter states do not accept a zero initialization seed")
    plan = build_measurement_plan(
        state_id=args.state_id,
        candidate_ids=candidate_ids,
        probe_seeds=args.probe_seeds,
        existing_rows=(),
    )
    if args.state_id != ZERO_STATE and args.adapter_dir is not None:
        _validate_adapter_dir(
            adapter_dir=args.adapter_dir.resolve(),
            state_id=args.state_id,
            adapter_index=adapter_index,
        )
    contract = {
        "schema_version": "candidate-utility-state-probe-run-contract-v3",
        "measurement_semantics": "state_conditioned_local_utility_fresh_adamw",
        "state_id": args.state_id,
        "zero_adapter_initialization_seed": args.zero_adapter_initialization_seed,
        "probe_seeds": [int(seed) for seed in args.probe_seeds],
        "historical_measurements_reused": False,
        "candidate_ids": candidate_ids,
        "candidate_limit": args.candidate_limit,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "learning_rate": 0.0002,
        "optimizer": "AdamW_fresh_zero_moments_per_probe",
        "cleanup_interval": args.cleanup_interval,
        "protocol_sha256": file_sha256(protocol_path),
        "preflight_sha256": file_sha256(preflight_path),
        "panel_sha256": file_sha256(panel_path),
        "adapter_index_sha256": file_sha256(adapter_index_path),
        "public_config_sha256": file_sha256(public_config_path),
        "plan": plan,
    }
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "gpu_accessed": False,
                    "state_id": args.state_id,
                    "new_measurement_count": plan["new_measurement_count"],
                    "reused_measurement_count": plan["reused_measurement_count"],
                    "contract_sha256": hashlib.sha256(
                        json.dumps(contract, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                },
                sort_keys=True,
            )
        )
        return
    if args.run_dir is None:
        raise ValueError("GPU execution requires --run-dir")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("state probe requires a BF16 CUDA GPU")
    run_dir = args.run_dir.resolve()
    contract_path = run_dir / "run_contract.json"
    measurement_path = run_dir / "utility_measurements.jsonl"
    if run_dir.exists():
        if not args.resume:
            raise FileExistsError(f"run directory exists without --resume: {run_dir}")
        if read_json(contract_path) != contract:
            raise ValueError("resume contract changed")
    else:
        if args.resume:
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(contract_path, contract)
    resume_rows = read_jsonl(measurement_path)
    completed = validate_resume_rows(plan=plan, rows=resume_rows)
    pending = [
        row
        for row in plan["new_measurements"]
        if measurement_key(row["state_id"], row["candidate_id"], row["probe_seed"])
        not in completed
    ]
    if not pending:
        raise ValueError("no pending measurements; run is already complete or plan is empty")

    config = read_json(public_config_path)
    model_config = config["model"]
    gsm_config = config["datasets"]["gsm8k"]
    tulu_config = config["datasets"]["candidate_pool"]
    gsm_records = sorted(
        (
            row
            for row in read_jsonl(data_dir / "gsm8k_records.jsonl")
            if row["protocol_split"] == "candidate_utility_validation"
        ),
        key=lambda row: (int(row["source_index"]), str(row["record_id"])),
    )
    if len(gsm_records) != 128:
        raise ValueError("utility set no longer contains 128 records")
    tulu_pool = {
        str(row["candidate_id"]): row
        for row in read_jsonl(data_dir / "tulu_candidate_pool.jsonl")
    }
    gsm = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        split="train",
        revision=gsm_config["revision"],
    )
    tulu = load_dataset(
        tulu_config["repo_id"],
        tulu_config["config"],
        split="train",
        revision=tulu_config["revision"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repo_id"], revision=model_config["revision"], use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer has no EOS token")
    collator = ResponseOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))
    utility_source_rows = [gsm[int(record["source_index"])] for record in gsm_records]
    for record, source_row in zip(gsm_records, utility_source_rows, strict=True):
        validate_gsm8k_source_row(record, source_row)
    utility_examples, _ = build_tokenized_overfit_examples(
        tokenizer=tokenizer,
        rows=utility_source_rows,
        record_ids=[str(record["record_id"]) for record in gsm_records],
        max_length=args.max_length,
    )
    utility_loader = DataLoader(
        utility_examples,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    probe_loader = DataLoader(
        utility_examples[: args.eval_batch_size],
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    candidate_examples: dict[str, dict[str, list[int]]] = {}
    for panel_row in candidate_rows:
        candidate_id = str(panel_row["candidate_id"])
        candidate = tulu_pool[candidate_id]
        messages = _validate_candidate(candidate, dict(tulu[int(candidate["source_index"])]))
        prompt, response = tulu_response_only_parts(messages, eos_token=tokenizer.eos_token)
        candidate_examples[candidate_id] = tokenize_response_only(
            tokenizer,
            prompt=prompt,
            response=response,
            max_length=args.max_length,
            add_eos=True,
        )

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    if args.zero_adapter_initialization_seed is not None:
        set_seed(args.zero_adapter_initialization_seed)
    torch.cuda.reset_peak_memory_stats()
    model = _load_state_model(
        repo_id=model_config["repo_id"],
        revision=model_config["revision"],
        state_id=args.state_id,
        adapter_dir=args.adapter_dir.resolve() if args.adapter_dir else None,
        device=device,
    )
    parameter_report = audit_lora_parameters(model)
    snapshot = _adapter_snapshot(model)
    state_snapshot_sha256 = _snapshot_sha256(snapshot)
    state_utility_loss = mean_supervised_token_loss(model, utility_loader, device)
    state_probe_loss = mean_supervised_token_loss(model, probe_loader, device)
    started = time.perf_counter()
    gradient_audited = False
    mode = "a" if measurement_path.exists() else "x"
    with measurement_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for plan_row in pending:
            candidate_id = str(plan_row["candidate_id"])
            probe_seed = int(plan_row["probe_seed"])
            set_seed(probe_seed)
            _restore_adapter(model, snapshot)
            initial_probe_loss = mean_supervised_token_loss(model, probe_loader, device)
            restore_difference = abs(initial_probe_loss - state_probe_loss)
            if restore_difference > 1e-7:
                raise RuntimeError(
                    f"adapter restore changed probe loss by {restore_difference}"
                )
            candidate_loader = DataLoader(
                [candidate_examples[candidate_id]],
                batch_size=1,
                shuffle=False,
                collate_fn=collator,
            )
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=0.0002, weight_decay=0.0)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = to_device(next(iter(candidate_loader)), device)
            train_loss = model(**batch).loss
            train_loss.backward()
            if not gradient_audited:
                audit_lora_gradients(model)
                gradient_audited = True
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            post_loss = mean_supervised_token_loss(model, utility_loader, device)
            row = {
                "status": "PASS",
                "state_id": args.state_id,
                "candidate_id": candidate_id,
                "probe_seed": probe_seed,
                "zero_adapter_initialization_seed": args.zero_adapter_initialization_seed,
                "state_snapshot_sha256": state_snapshot_sha256,
                "state_utility_loss": state_utility_loss,
                "post_utility_loss": post_loss,
                "utility": state_utility_loss - post_loss,
                "candidate_train_loss": float(train_loss.item()),
                "gradient_norm_before_clipping": gradient_norm,
                "restore_probe_loss_difference": restore_difference,
                "trainable_parameters": parameter_report.trainable_parameters,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed.add(measurement_key(args.state_id, candidate_id, probe_seed))
            print(
                f"completed={len(completed)}/{plan['new_measurement_count']} "
                f"state={args.state_id} candidate={candidate_id} seed={probe_seed}",
                flush=True,
            )
            del optimizer, trainable, batch, candidate_loader
            if len(completed) % args.cleanup_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()
    final_rows = read_jsonl(measurement_path)
    validate_resume_rows(plan=plan, rows=final_rows)
    if len(final_rows) != plan["new_measurement_count"]:
        raise ValueError("run ended without all frozen measurements")
    complete = {
        "schema_version": "candidate-utility-state-probe-complete-v3",
        "status": "PASS",
        "state_id": args.state_id,
        "new_measurement_count": len(final_rows),
        "reused_measurement_count": 0,
        "state_snapshot_sha256": state_snapshot_sha256,
        "state_utility_loss": state_utility_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "measurement_sha256": file_sha256(measurement_path),
    }
    write_json_exclusive(run_dir / "COMPLETE.json", complete)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
