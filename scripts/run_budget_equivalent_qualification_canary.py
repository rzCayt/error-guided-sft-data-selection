"""Run the fixed 128-row GSM8K canary for a qualification adapter."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.cloud_v2_batching import append_jsonl_rows_fsynced  # noqa: E402
from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.evaluation.resumable import (  # noqa: E402
    aggregate_gsm8k_metrics,
    validate_completed_prefix,
)
from eg_sft.experiment.budget_equivalent_qualification import (  # noqa: E402
    resolve_qualification_contract,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


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
        raise RuntimeError("qualification canary requires a BF16 CUDA GPU")
    contract = resolve_qualification_contract(
        repo_root=ROOT,
        qualification_config_path=args.qualification_config.resolve(),
    )
    run_dir = args.overfit_run_dir.resolve()
    metrics = _read_json(run_dir / "metrics.json")
    gates = contract["qualification"]["single_gpu_gates"]
    if not bool(metrics.get("overfit_gate_loss_ratio_at_most_0_5")):
        raise ValueError("qualification canary requires a passed overfit gate")
    if not bool(metrics.get("adapter_reload_gate_difference_at_most_1e_6")):
        raise ValueError("qualification canary requires a passed adapter reload gate")

    output_dir = run_dir / "qualification" / "canary128"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_outputs.jsonl"
    metrics_path = output_dir / "sealed_metrics.json"
    manifest_path = output_dir / "manifest.json"
    adapter_dir = run_dir / "adapter"
    adapter_sha256 = file_sha256(adapter_dir / "adapter_model.safetensors")
    manifest = {
        "schema_version": "budget-equivalent-qualification-canary-v1",
        "qualification_config_sha256": contract["qualification_config_sha256"],
        "matrix_config_sha256": contract["matrix_sha256"],
        "adapter_model_sha256": adapter_sha256,
        "record_count": len(contract["canary_records"]),
        "record_id_sha256": __import__("hashlib").sha256(
            ("\n".join(str(row["record_id"]) for row in contract["canary_records"]) + "\n").encode()
        ).hexdigest(),
        "accuracy_withheld": True,
    }
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError("qualification canary manifest changed")
    else:
        _write_json_exclusive(manifest_path, manifest)
    completed = read_jsonl(raw_path) if raw_path.exists() else []
    validate_completed_prefix(
        completed_rows=completed,
        frozen_records=contract["canary_records"],
    )
    if metrics_path.exists():
        if len(completed) != int(gates["canary_output_count"]):
            raise ValueError("canary metrics exist before 128 outputs")
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "stage": "budget_equivalent_qualification_canary",
                    "record_count": len(completed),
                    "accuracy_withheld": True,
                    "raw_outputs_sha256": file_sha256(raw_path),
                },
                sort_keys=True,
            )
        )
        return

    protocol = contract["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        protocol["model"]["repo_id"],
        revision=protocol["model"]["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    device = torch.device("cuda")
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to(device)
    model.config.use_cache = True
    model.eval()
    set_seed(int(protocol["seed"]))
    gsm_spec = protocol["datasets"]["gsm8k"]
    gsm_test = load_dataset(
        gsm_spec["repo_id"],
        gsm_spec["config"],
        split="test",
        revision=gsm_spec["revision"],
    )
    evaluation = contract["matrix"]["evaluation"]
    for index in range(len(completed), len(contract["canary_records"])):
        record = contract["canary_records"][index]
        source_row = gsm_test[int(record["source_index"])]
        validate_gsm8k_source_row(record, source_row)
        prompt = build_evaluation_prompt(source_row["question"])
        encoded = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(evaluation["max_input_length"]),
        ).to(device)
        width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=int(evaluation["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw_output = tokenizer.decode(generated[0, width:], skip_special_tokens=True).strip()
        row = score_generation(
            record=record,
            gold_answer_text=source_row["answer"],
            generated_text=raw_output,
        )
        row["canary_index"] = index
        append_jsonl_rows_fsynced(raw_path, [row])
        if (index + 1) % 16 == 0 or index + 1 == len(contract["canary_records"]):
            print(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "stage": "budget_equivalent_qualification_canary",
                        "progress": f"{index + 1}/{len(contract['canary_records'])}",
                        "accuracy_withheld": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    rows = read_jsonl(raw_path)
    validate_completed_prefix(
        completed_rows=rows,
        frozen_records=contract["canary_records"],
    )
    aggregate = aggregate_gsm8k_metrics(rows)
    sealed = {
        **aggregate,
        "status": "PASS",
        "record_count": len(rows),
        "raw_outputs_sha256": file_sha256(raw_path),
        "adapter_model_sha256": adapter_sha256,
        "accuracy_withheld": True,
    }
    _write_json_exclusive(metrics_path, sealed)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "stage": "budget_equivalent_qualification_canary",
                "record_count": len(rows),
                "accuracy_withheld": True,
                "raw_outputs_sha256": file_sha256(raw_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
