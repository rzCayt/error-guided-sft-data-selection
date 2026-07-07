from __future__ import annotations

import json
import sys
from pathlib import Path


def write_no_training_evidence(output_dir: Path, reason: str, model: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "not_run",
        "model": model,
        "reason": reason,
        "python": sys.version,
        "next_steps": [
            "Install train extras: python -m pip install -e .[train]",
            "Run on a GPU machine with enough memory for Qwen/Qwen2.5-0.5B.",
            "Save only logs and metrics; do not commit adapter checkpoints.",
        ],
    }
    path = output_dir / "no_training_evidence.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_lora_smoke(model: str, output_dir: Path) -> Path:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return write_no_training_evidence(output_dir, f"Training dependencies unavailable: {exc}", model)

    return write_no_training_evidence(
        output_dir,
        "Dependencies are installed, but full LoRA training is intentionally not launched by the scaffold.",
        model,
    )
