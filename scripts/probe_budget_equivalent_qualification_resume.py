"""Exercise adapter, optimizer, scheduler, and RNG checkpoint restore on one real batch."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import _restore_rng_state, _rng_state  # noqa: E402
from run_gsm8k_lora_overfit import _read_development_records  # noqa: E402

from eg_sft.experiment.budget_equivalent_qualification import (  # noqa: E402
    resolve_qualification_contract,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402
from eg_sft.training.overfit import build_tokenized_overfit_examples  # noqa: E402
from eg_sft.training.response_only import ResponseOnlyCollator  # noqa: E402


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _model(protocol: dict[str, Any], adapter_dir: Path, device: torch.device) -> PeftModel:
    base = AutoModelForCausalLM.from_pretrained(
        protocol["model"]["repo_id"],
        revision=protocol["model"]["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True).to(device)
    model.config.use_cache = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qualification-config",
        type=Path,
        default=Path("configs/budget_equivalent_qualification_v2.json"),
    )
    parser.add_argument("--overfit-run-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("qualification resume probe requires a BF16 CUDA GPU")

    contract = resolve_qualification_contract(
        repo_root=ROOT,
        qualification_config_path=args.qualification_config.resolve(),
    )
    run_dir = args.overfit_run_dir.resolve()
    output_dir = run_dir / "qualification" / "resume_probe"
    report_path = output_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise ValueError("existing qualification resume probe did not pass")
        print(json.dumps(report, sort_keys=True))
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = contract["protocol"]
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    record = _read_development_records(
        contract["data_dir"] / "gsm8k_records.jsonl", 1
    )[0]
    gsm_spec = protocol["datasets"]["gsm8k"]
    gsm_train = load_dataset(
        gsm_spec["repo_id"],
        gsm_spec["config"],
        split="train",
        revision=gsm_spec["revision"],
    )
    examples, _ = build_tokenized_overfit_examples(
        tokenizer=tokenizer,
        rows=[gsm_train[int(record["source_index"])]],
        record_ids=[str(record["record_id"])],
        max_length=512,
    )
    collator = ResponseOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))
    batch = {name: value.to(device) for name, value in collator(examples).items()}
    adapter_dir = run_dir / "adapter"
    model = _model(protocol, adapter_dir, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=2
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        reference_loss = float(model(**batch).loss.item())
    checkpoint = {
        "adapter_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in get_peft_model_state_dict(model).items()
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": _rng_state(),
        "binding": {
            "qualification_config_sha256": contract["qualification_config_sha256"],
            "matrix_config_sha256": contract["matrix_sha256"],
            "adapter_model_sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
            "record_id": record["record_id"],
        },
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    expected_random = torch.rand(8, device=device).cpu()
    original_optimizer_state = optimizer.state_dict()
    original_scheduler_state = scheduler.state_dict()
    del model, optimizer, scheduler, trainable
    gc.collect()
    torch.cuda.empty_cache()

    restored_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = _model(protocol, adapter_dir, device)
    set_peft_model_state_dict(restored, restored_checkpoint["adapter_state"])
    restored_trainable = [
        parameter for parameter in restored.parameters() if parameter.requires_grad
    ]
    restored_optimizer = torch.optim.AdamW(restored_trainable, lr=2e-4, weight_decay=0.0)
    restored_scheduler = get_linear_schedule_with_warmup(
        restored_optimizer, num_warmup_steps=0, num_training_steps=2
    )
    restored_optimizer.load_state_dict(restored_checkpoint["optimizer_state"])
    restored_scheduler.load_state_dict(restored_checkpoint["scheduler_state"])
    _restore_rng_state(restored_checkpoint["rng_state"])
    observed_random = torch.rand(8, device=device).cpu()
    restored.eval()
    with torch.no_grad():
        restored_loss = float(restored(**batch).loss.item())
    loss_difference = abs(reference_loss - restored_loss)
    optimizer_equal = _nested_equal(
        original_optimizer_state, restored_optimizer.state_dict()
    )
    scheduler_equal = _nested_equal(
        original_scheduler_state, restored_scheduler.state_dict()
    )
    rng_equal = bool(torch.equal(expected_random, observed_random))
    status = (
        "PASS"
        if loss_difference <= 1e-6 and optimizer_equal and scheduler_equal and rng_equal
        else "FAIL"
    )
    report = {
        "schema_version": "budget-equivalent-qualification-resume-probe-v1",
        "status": status,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "adapter_state_restored": loss_difference <= 1e-6,
        "adapter_reload_loss_absolute_difference": loss_difference,
        "optimizer_state_restored": optimizer_equal,
        "scheduler_state_restored": scheduler_equal,
        "rng_state_restored": rng_equal,
        "formal_phase1_selection_consumed": False,
        "claim_boundary": "Checkpoint mechanics only; no selector evidence.",
    }
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    if status != "PASS":
        raise RuntimeError("qualification resume probe failed")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
