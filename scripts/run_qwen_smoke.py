from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def extract_numbers(text: str) -> list[float]:
    return [float(match) for match in NUMBER_RE.findall(text.replace(",", ""))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--prompt",
        default="Problem: A metric starts at 100 and increases by 15%. Final value =",
    )
    parser.add_argument("--expected", type=float, default=115.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", default="results/qwen2_5_0_5b_smoke.json")
    args = parser.parse_args()

    payload: dict = {
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prompt": args.prompt,
        "expected": args.expected,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "python": sys.version,
        "platform": platform.platform(),
    }

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        payload.update(
            {
                "status": "not_run",
                "reason": f"Required inference dependencies unavailable: {exc}",
            }
        )
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload["packages"] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    payload["device"] = {
        "type": device,
        "name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    except TypeError:  # pragma: no cover - older transformers compatibility
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_len:]
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    numbers = extract_numbers(continuation)
    first_number = numbers[0] if numbers else None

    memory: dict[str, float] = {}
    if device == "cuda":
        memory = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        }

    payload.update(
        {
            "status": "ok",
            "model_revision": getattr(model.config, "_commit_hash", None),
            "raw_continuation": continuation,
            "full_text": full_text,
            "continuation_numbers": numbers,
            "first_continuation_number": first_number,
            "first_number_matches_expected": first_number is not None
            and abs(first_number - args.expected) <= 1e-6,
            "cuda_memory": memory,
            "note": (
                "This smoke only proves that the base model loads and can produce a short "
                "completion on this machine. It is not a full diagnostic or LoRA result."
            ),
        }
    )

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
