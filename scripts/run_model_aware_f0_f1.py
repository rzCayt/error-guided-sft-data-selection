from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.utils.io import read_jsonl  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DTYPE = "bfloat16"
RUN_NAME = "qwen3_1_7b_exact_chat_max192"
PARSER_V3_VERSION = "parse_numeric_final_marker_only_v4"
TASK_FAMILY = "weighted_aggregation"
MAX_SEQUENCE_TOKENS = 256

CANDIDATE_PATH = ROOT / "data/samples/candidate_pool.jsonl"
DEV_PATH = ROOT / "data/samples/dev_diagnostic.jsonl"
ERROR_MEMBERSHIP_PATH = (
    ROOT / "results/strong_baseline_protocol_v2_ab_rescore/parser_v3_outputs.jsonl"
)
PRIOR_INFERENCE_METADATA_PATH = (
    ROOT
    / "results/strong_baseline_protocol_v2_ab/qwen3_1_7b_exact_chat_max192/"
    "scale_model_diagnostic_run_metadata.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/model_aware_signal_f0_f1"

EXPECTED_CANDIDATE_COUNT = 125
EXPECTED_DEV_COUNT = 25
EXPECTED_ERROR_COUNT = 17
EXPECTED_CORRECT_COUNT = 8

BACKWARD_RESERVE_MIB = 2048.0
FRAMEWORK_SAFETY_RESERVE_MIB = 512.0
F0_ESTIMATE_LIMIT_MIB = 6656.0
F1_PREFLIGHT_FREE_MIB = 7680.0
F1_ALLOCATED_LIMIT_MIB = 7168.0
F1_RESERVED_LIMIT_MIB = 7680.0
MIN_GRADIENT_NORM = 1e-12

FORBIDDEN_OUTPUT_KEY_PARTS = (
    "cosine",
    "ranking",
    "rank",
    "subset",
    "top_k",
    "topk",
    "research_score",
    "gradient_vector",
    "dot_product",
)
ALLOWED_INPUT_PATHS = {
    CANDIDATE_PATH.resolve(),
    DEV_PATH.resolve(),
    ERROR_MEMBERSHIP_PATH.resolve(),
    PRIOR_INFERENCE_METADATA_PATH.resolve(),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_answer(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def content_hash(row: dict[str, Any]) -> str:
    payload = f"{row['prompt']}{row['rationale']}{canonical_answer(row['answer'])}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def target_text(row: dict[str, Any]) -> str:
    return f"{row['rationale']}\nFinal answer: {canonical_answer(row['answer'])}"


def validate_input_allowlist(paths: list[Path]) -> None:
    resolved = {path.resolve() for path in paths}
    if resolved - ALLOWED_INPUT_PATHS:
        raise ValueError(f"input path is outside the frozen allowlist: {sorted(resolved - ALLOWED_INPUT_PATHS)}")
    forbidden_markers = ("pair_manifest", "hidden_group", "test_id", "test_ood")
    for path in resolved:
        lowered = str(path).lower()
        if any(marker in lowered for marker in forbidden_markers):
            raise ValueError(f"forbidden input path: {path}")


def _as_token_ids(value: object) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[assignment, union-attr]
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("chat template must return a token id list")
    return [int(token_id) for token_id in value]


def serialize_teacher_forcing(tokenizer: object, row: dict[str, Any]) -> dict[str, Any]:
    user_messages = [{"role": "user", "content": row["prompt"]}]
    full_messages = user_messages + [{"role": "assistant", "content": target_text(row)}]
    prefix_ids = _as_token_ids(
        tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full_ids = _as_token_ids(
        tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("full chat serialization does not begin with the generation prefix")
    target_token_count = len(full_ids) - len(prefix_ids)
    if target_token_count <= 0:
        raise ValueError("assistant target has no trainable tokens")
    if len(full_ids) > MAX_SEQUENCE_TOKENS:
        raise ValueError(
            f"serialized row exceeds {MAX_SEQUENCE_TOKENS} tokens: {row['id']}={len(full_ids)}"
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :],
        "prefix_token_count": len(prefix_ids),
        "target_token_count": target_token_count,
        "total_token_count": len(full_ids),
        "content_sha256": content_hash(row),
    }


def load_frozen_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validate_input_allowlist([CANDIDATE_PATH, DEV_PATH, ERROR_MEMBERSHIP_PATH])
    candidates = [
        row
        for row in read_jsonl(CANDIDATE_PATH)
        if row.get("split") == "candidate_pool" and row.get("task_family") == TASK_FAMILY
    ]
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CANDIDATE_COUNT} weighted candidates, found {len(candidates)}"
        )

    membership_rows = [
        row
        for row in read_jsonl(ERROR_MEMBERSHIP_PATH)
        if row.get("run") == RUN_NAME
        and row.get("model") == MODEL
        and row.get("model_revision") == REVISION
        and row.get("task_family") == TASK_FAMILY
        and row.get("parser_v3_version") == PARSER_V3_VERSION
    ]
    if len(membership_rows) != EXPECTED_DEV_COUNT:
        raise ValueError(f"expected {EXPECTED_DEV_COUNT} frozen dev rows, found {len(membership_rows)}")
    error_ids = {
        str(row["id"]).split("::", 1)[0]
        for row in membership_rows
        if not bool(row["parser_v3_correct"])
    }
    correct_ids = {
        str(row["id"]).split("::", 1)[0]
        for row in membership_rows
        if bool(row["parser_v3_correct"])
    }
    if len(error_ids) != EXPECTED_ERROR_COUNT or len(correct_ids) != EXPECTED_CORRECT_COUNT:
        raise ValueError(
            "frozen error membership mismatch: "
            f"errors={len(error_ids)}, correct={len(correct_ids)}"
        )
    if error_ids & correct_ids:
        raise ValueError("a dev id appears in both error and correct membership")

    dev_by_id = {
        str(row["id"]): row
        for row in read_jsonl(DEV_PATH)
        if row.get("split") == "dev_diagnostic" and row.get("task_family") == TASK_FAMILY
    }
    missing = (error_ids | correct_ids) - set(dev_by_id)
    if missing:
        raise ValueError(f"membership ids missing from dev data: {sorted(missing)}")
    errors = [dev_by_id[row_id] for row_id in sorted(error_ids)]
    correct = [dev_by_id[row_id] for row_id in sorted(correct_ids)]
    return candidates, errors, correct


def choose_longest(
    tokenizer: object,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    serialized = [(row, serialize_teacher_forcing(tokenizer, row)) for row in rows]
    max_tokens = max(info["total_token_count"] for _, info in serialized)
    max_length = [(row, info) for row, info in serialized if info["total_token_count"] == max_tokens]
    chosen_hash = min(info["content_sha256"] for _, info in max_length)
    equivalent = [(row, info) for row, info in max_length if info["content_sha256"] == chosen_hash]
    row, info = equivalent[0]
    equivalent_ids = sorted(str(item[0]["id"]) for item in equivalent)
    return row, info, equivalent_ids


def parameter_spec_from_config(config: object) -> dict[str, Any]:
    num_layers = int(getattr(config, "num_hidden_layers"))
    hidden_size = int(getattr(config, "hidden_size"))
    num_heads = int(getattr(config, "num_attention_heads"))
    num_kv_heads = int(getattr(config, "num_key_value_heads"))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
    last_layer = num_layers - 1
    specs = [
        {
            "name": f"model.layers.{last_layer}.self_attn.q_proj.weight",
            "shape": [num_heads * head_dim, hidden_size],
        },
        {
            "name": f"model.layers.{last_layer}.self_attn.v_proj.weight",
            "shape": [num_kv_heads * head_dim, hidden_size],
        },
    ]
    for spec in specs:
        spec["parameter_count"] = math.prod(spec["shape"])
        spec["fp32_gradient_bytes"] = spec["parameter_count"] * 4
    total_parameters = sum(int(spec["parameter_count"]) for spec in specs)
    gradient_bytes = total_parameters * 4
    return {
        "last_layer_index": last_layer,
        "parameters": specs,
        "total_parameter_count": total_parameters,
        "fp32_gradient_bytes": gradient_bytes,
        "fp32_gradient_mib": gradient_bytes / 1024**2,
    }


def estimate_peak_mib(prior_peak_mib: float, gradient_mib: float) -> float:
    return prior_peak_mib + gradient_mib + BACKWARD_RESERVE_MIB + FRAMEWORK_SAFETY_RESERVE_MIB


def sample_metadata(row: dict[str, Any], info: dict[str, Any], equivalent_ids: list[str]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "content_sha256": info["content_sha256"],
        "equivalent_content_ids": equivalent_ids,
        "prefix_token_count": info["prefix_token_count"],
        "target_token_count": info["target_token_count"],
        "total_token_count": info["total_token_count"],
    }


def run_f0() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        import transformers
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"F0 dependencies unavailable: {exc}") from exc

    validate_input_allowlist(
        [CANDIDATE_PATH, DEV_PATH, ERROR_MEMBERSHIP_PATH, PRIOR_INFERENCE_METADATA_PATH]
    )
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    config_revision = getattr(config, "_commit_hash", None)
    if config_revision != REVISION:
        raise ValueError(f"config revision mismatch: expected {REVISION}, got {config_revision}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or (
        getattr(tokenizer, "init_kwargs", {}) or {}
    ).get("_commit_hash")
    if tokenizer_revision not in {None, REVISION}:
        raise ValueError(f"tokenizer revision mismatch: expected {REVISION}, got {tokenizer_revision}")

    candidates, errors, correct = load_frozen_rows()
    candidate_row, candidate_info, candidate_equivalent = choose_longest(tokenizer, candidates)
    error_row, error_info, error_equivalent = choose_longest(tokenizer, errors)
    parameter_spec = parameter_spec_from_config(config)

    prior_metadata = json.loads(PRIOR_INFERENCE_METADATA_PATH.read_text(encoding="utf-8"))
    if prior_metadata.get("model_revision") != REVISION:
        raise ValueError("prior inference metadata revision mismatch")
    prior_peak_mib = float(prior_metadata["cuda_memory"]["peak_allocated_mb"])
    estimated_peak_mib = estimate_peak_mib(
        prior_peak_mib, float(parameter_spec["fp32_gradient_mib"])
    )
    status = "passed" if estimated_peak_mib <= F0_ESTIMATE_LIMIT_MIB else "blocked"
    artifact = {
        "schema_version": 1,
        "stage_id": "model_aware_signal_f0_f1",
        "phase": "f0_static_estimate",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "revision": REVISION,
        "dtype": DTYPE,
        "packages": {"transformers": transformers.__version__},
        "config": {
            "model_type": getattr(config, "model_type", None),
            "num_hidden_layers": int(getattr(config, "num_hidden_layers")),
            "hidden_size": int(getattr(config, "hidden_size")),
            "num_attention_heads": int(getattr(config, "num_attention_heads")),
            "num_key_value_heads": int(getattr(config, "num_key_value_heads")),
            "config_revision": config_revision,
            "tokenizer_revision": tokenizer_revision or REVISION,
        },
        "input_hashes": {
            "candidate_pool_sha256": sha256_file(CANDIDATE_PATH),
            "dev_diagnostic_sha256": sha256_file(DEV_PATH),
            "error_membership_sha256": sha256_file(ERROR_MEMBERSHIP_PATH),
            "prior_inference_metadata_sha256": sha256_file(PRIOR_INFERENCE_METADATA_PATH),
        },
        "validated_counts": {
            "weighted_candidates": len(candidates),
            "weighted_dev": len(errors) + len(correct),
            "error_queries": len(errors),
            "correct_queries": len(correct),
        },
        "parameter_spec": parameter_spec,
        "longest_candidate": sample_metadata(
            candidate_row, candidate_info, candidate_equivalent
        ),
        "longest_error_query": sample_metadata(error_row, error_info, error_equivalent),
        "memory_estimate_mib": {
            "prior_inference_peak_allocated": prior_peak_mib,
            "selected_fp32_gradient": round(float(parameter_spec["fp32_gradient_mib"]), 6),
            "backward_reserve": BACKWARD_RESERVE_MIB,
            "framework_safety_reserve": FRAMEWORK_SAFETY_RESERVE_MIB,
            "estimated_peak": round(estimated_peak_mib, 6),
            "f0_limit": F0_ESTIMATE_LIMIT_MIB,
        },
        "f1_allowed": status == "passed",
        "claim_boundary": "F0 is a static engineering estimate, not evidence of signal validity.",
    }
    return artifact, candidate_row, error_row


def run_correct_query_length_audit(
    static_artifact: dict[str, Any], static_artifact_path: Path
) -> dict[str, Any]:
    try:
        import transformers
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"correct-query audit dependencies unavailable: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or (
        getattr(tokenizer, "init_kwargs", {}) or {}
    ).get("_commit_hash")
    if tokenizer_revision not in {None, REVISION}:
        raise ValueError(f"tokenizer revision mismatch: expected {REVISION}, got {tokenizer_revision}")
    _, _, correct = load_frozen_rows()
    correct_row, correct_info, equivalent_ids = choose_longest(tokenizer, correct)
    previously_smoked_max = max(
        int(static_artifact["longest_candidate"]["total_token_count"]),
        int(static_artifact["longest_error_query"]["total_token_count"]),
    )
    backward_required = int(correct_info["total_token_count"]) > previously_smoked_max
    return {
        "schema_version": 1,
        "stage_id": "model_aware_signal_f0_f1",
        "phase": "f0_correct_query_length_audit",
        "status": "blocked" if backward_required else "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "revision": REVISION,
        "tokenizer_revision": tokenizer_revision or REVISION,
        "packages": {"transformers": transformers.__version__},
        "correct_query_count": len(correct),
        "longest_correct_query": sample_metadata(
            correct_row, correct_info, equivalent_ids
        ),
        "previously_smoked_max_total_tokens": previously_smoked_max,
        "correct_query_exceeds_smoked_max": backward_required,
        "additional_backward_required": backward_required,
        "input_hashes": {
            "dev_diagnostic_sha256": sha256_file(DEV_PATH),
            "error_membership_sha256": sha256_file(ERROR_MEMBERSHIP_PATH),
            "static_estimate_sha256": sha256_file(static_artifact_path),
        },
        "claim_boundary": (
            "This supplement only closes F2 input-length coverage; it does not measure "
            "gradient-signal validity."
        ),
    }


def ensure_no_forbidden_output_keys(value: object, prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in FORBIDDEN_OUTPUT_KEY_PARTS):
                raise ValueError(f"forbidden output key: {prefix}{key}")
            ensure_no_forbidden_output_keys(item, f"{prefix}{key}.")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            ensure_no_forbidden_output_keys(item, f"{prefix}{index}.")


def _mib(value: int) -> float:
    return round(value / 1024**2, 3)


def query_nvidia_smi_memory_mib() -> dict[str, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    first_line = next(line for line in completed.stdout.splitlines() if line.strip())
    values = [float(value.strip()) for value in first_line.split(",")]
    if len(values) != 3:
        raise ValueError(f"unexpected nvidia-smi memory output: {first_line}")
    return {"total": values[0], "used": values[1], "free": values[2]}


def configure_selected_parameters(model: object, selected_parameter_names: list[str]) -> list[str]:
    for parameter in model.parameters():  # type: ignore[attr-defined]
        parameter.requires_grad_(False)
    named_parameters = dict(model.named_parameters())  # type: ignore[attr-defined]
    missing_parameters = [name for name in selected_parameter_names if name not in named_parameters]
    if missing_parameters:
        raise ValueError(f"selected parameters missing from model: {missing_parameters}")
    for name in selected_parameter_names:
        named_parameters[name].requires_grad_(True)
    trainable_names = [
        name
        for name, parameter in model.named_parameters()  # type: ignore[attr-defined]
        if parameter.requires_grad
    ]
    if trainable_names != selected_parameter_names:
        raise ValueError(f"unexpected trainable parameters: {trainable_names}")
    return trainable_names


def run_single_backward(
    torch: object,
    model: object,
    tokenizer: object,
    row: dict[str, Any],
    kind: str,
    selected_parameter_names: list[str],
) -> dict[str, Any]:
    serialized = serialize_teacher_forcing(tokenizer, row)
    model.zero_grad(set_to_none=True)  # type: ignore[attr-defined]
    gc.collect()
    torch.cuda.empty_cache()  # type: ignore[attr-defined]
    torch.cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]
    torch.cuda.synchronize()  # type: ignore[attr-defined]

    input_ids = torch.tensor(  # type: ignore[attr-defined]
        [serialized["input_ids"]], dtype=torch.long, device="cuda"  # type: ignore[attr-defined]
    )
    attention_mask = torch.tensor(  # type: ignore[attr-defined]
        [serialized["attention_mask"]], dtype=torch.long, device="cuda"  # type: ignore[attr-defined]
    )
    labels = torch.tensor(  # type: ignore[attr-defined]
        [serialized["labels"]], dtype=torch.long, device="cuda"  # type: ignore[attr-defined]
    )
    started = time.perf_counter()
    outputs = model(  # type: ignore[operator]
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss
    loss_finite = bool(torch.isfinite(loss).item())  # type: ignore[attr-defined]
    if loss_finite:
        loss.backward()
    torch.cuda.synchronize()  # type: ignore[attr-defined]
    elapsed_seconds = time.perf_counter() - started

    named_parameters = dict(model.named_parameters())  # type: ignore[attr-defined]
    gradients = [named_parameters[name].grad for name in selected_parameter_names]
    all_gradients_present = all(gradient is not None for gradient in gradients)
    gradient_finite = all(
        gradient is not None and bool(torch.isfinite(gradient).all().item())  # type: ignore[attr-defined]
        for gradient in gradients
    )
    norm_squared = 0.0
    if gradient_finite:
        for gradient in gradients:
            norm = torch.linalg.vector_norm(gradient.float()).item()  # type: ignore[union-attr,attr-defined]
            norm_squared += float(norm) ** 2
    gradient_nonzero = math.sqrt(norm_squared) >= MIN_GRADIENT_NORM
    unexpected_gradients = [
        name
        for name, parameter in named_parameters.items()
        if name not in selected_parameter_names and parameter.grad is not None
    ]
    peak_allocated_mib = _mib(torch.cuda.max_memory_allocated())  # type: ignore[attr-defined]
    peak_reserved_mib = _mib(torch.cuda.max_memory_reserved())  # type: ignore[attr-defined]

    hard_stop_reasons = []
    if not loss_finite:
        hard_stop_reasons.append("non_finite_loss")
    if not all_gradients_present:
        hard_stop_reasons.append("missing_selected_gradient")
    if not gradient_finite:
        hard_stop_reasons.append("non_finite_gradient")
    if not gradient_nonzero:
        hard_stop_reasons.append("gradient_norm_below_threshold")
    if unexpected_gradients:
        hard_stop_reasons.append("unexpected_parameter_gradient")
    if peak_allocated_mib > F1_ALLOCATED_LIMIT_MIB:
        hard_stop_reasons.append("peak_allocated_limit_exceeded")
    if peak_reserved_mib > F1_RESERVED_LIMIT_MIB:
        hard_stop_reasons.append("peak_reserved_limit_exceeded")

    result = {
        "kind": kind,
        "id": row["id"],
        "content_sha256": serialized["content_sha256"],
        "prefix_token_count": serialized["prefix_token_count"],
        "target_token_count": serialized["target_token_count"],
        "total_token_count": serialized["total_token_count"],
        "loss_finite": loss_finite,
        "all_selected_gradients_present": all_gradients_present,
        "gradient_finite": gradient_finite,
        "gradient_nonzero": gradient_nonzero,
        "unexpected_gradient_parameters": unexpected_gradients,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "status": "passed" if not hard_stop_reasons else "stopped",
        "hard_stop_reasons": hard_stop_reasons,
    }

    del outputs, loss, input_ids, attention_mask, labels
    model.zero_grad(set_to_none=True)  # type: ignore[attr-defined]
    gc.collect()
    torch.cuda.empty_cache()  # type: ignore[attr-defined]
    return result


def run_f1(
    static_artifact: dict[str, Any],
    candidate_row: dict[str, Any],
    error_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        external_gpu_memory = query_nvidia_smi_memory_mib()
    except Exception as exc:
        stopped = {
            "schema_version": 1,
            "stage_id": "model_aware_signal_f0_f1",
            "phase": "f1_memory_smoke",
            "status": "stopped",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "revision": REVISION,
            "dtype": DTYPE,
            "samples": [],
            "hard_stop_reasons": [f"nvidia_smi_preflight_failed:{type(exc).__name__}:{exc}"],
            "claim_boundary": "F1 stopped before CUDA initialization and produced no research score.",
        }
        return stopped, {"status": "stopped", "packages": {}}
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"F1 dependencies unavailable: {exc}") from exc

    base_metadata: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": "model_aware_signal_f0_f1",
        "phase": "f1_memory_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "revision": REVISION,
        "dtype": DTYPE,
        "samples": [],
        "hard_stop_reasons": [],
        "claim_boundary": "F1 measures engineering feasibility only and stores no research score.",
        "preflight_nvidia_smi_memory_mib": external_gpu_memory,
    }
    if not static_artifact.get("f1_allowed"):
        base_metadata.update(
            {"status": "not_run_due_f0", "hard_stop_reasons": ["f0_gate_failed"]}
        )
        return base_metadata, {"status": "blocked", "packages": {}}
    if external_gpu_memory["free"] < F1_PREFLIGHT_FREE_MIB:
        base_metadata.update(
            {"status": "stopped", "hard_stop_reasons": ["external_gpu_contention"]}
        )
        return base_metadata, {
            "status": "stopped",
            "packages": {"torch": torch.__version__, "transformers": transformers.__version__},
        }
    if not torch.cuda.is_available():
        base_metadata.update(
            {"status": "stopped", "hard_stop_reasons": ["cuda_unavailable"]}
        )
        return base_metadata, {"status": "stopped", "packages": {"torch": torch.__version__}}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except TypeError:  # pragma: no cover - transformers compatibility
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    model.to("cuda")
    model.eval()
    model.config.use_cache = False
    model_revision = getattr(model.config, "_commit_hash", None)
    if model_revision != REVISION:
        raise ValueError(f"loaded model revision mismatch: {model_revision}")
    if model.dtype != torch.bfloat16:
        raise ValueError(f"loaded model dtype mismatch: {model.dtype}")

    selected_parameter_names = [
        str(spec["name"])
        for spec in static_artifact["parameter_spec"]["parameters"]
    ]
    configure_selected_parameters(model, selected_parameter_names)

    try:
        for kind, row in (("longest_candidate", candidate_row), ("longest_error_query", error_row)):
            sample_result = run_single_backward(
                torch,
                model,
                tokenizer,
                row,
                kind,
                selected_parameter_names,
            )
            base_metadata["samples"].append(sample_result)
            if sample_result["status"] != "passed":
                base_metadata["hard_stop_reasons"].extend(sample_result["hard_stop_reasons"])
                break
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        base_metadata["hard_stop_reasons"].append("cuda_oom")

    base_metadata["status"] = "passed" if not base_metadata["hard_stop_reasons"] and len(base_metadata["samples"]) == 2 else "stopped"
    base_metadata["selected_parameter_names"] = selected_parameter_names
    run_details = {
        "status": base_metadata["status"],
        "packages": {"torch": torch.__version__, "transformers": transformers.__version__},
        "device": {"type": "cuda", "name": torch.cuda.get_device_name(0)},
        "model_revision": model_revision,
        "model_dtype": str(model.dtype),
        "selected_parameter_names": selected_parameter_names,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return base_metadata, run_details


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_no_forbidden_output_keys(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def prepare_output_dir(output_dir: Path, phase: str) -> None:
    static_path = output_dir / "static_estimate.json"
    memory_path = output_dir / "memory_smoke.json"
    correct_audit_path = output_dir / "correct_query_length_audit.json"
    if phase in {"all", "f0"} and output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty; refusing to overwrite: {output_dir}")
    if phase == "f1":
        if not static_path.exists():
            raise SystemExit(f"F1 requires an existing F0 artifact: {static_path}")
        if memory_path.exists():
            raise SystemExit(f"memory smoke already exists; refusing to overwrite: {memory_path}")
    if phase == "correct-audit":
        if not static_path.exists():
            raise SystemExit(f"correct-query audit requires F0 artifact: {static_path}")
        if correct_audit_path.exists():
            raise SystemExit(
                f"correct-query length audit already exists; refusing to overwrite: {correct_audit_path}"
            )


def find_row_by_hash(rows: list[dict[str, Any]], expected_hash: str) -> dict[str, Any]:
    matches = [row for row in rows if content_hash(row) == expected_hash]
    if not matches:
        raise ValueError(f"selected content hash not found: {expected_hash}")
    return matches[0]


def run_metadata_base(phase: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage_id": "model_aware_signal_f0_f1",
        "phase_requested": phase,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "revision": REVISION,
        "dtype": DTYPE,
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "training_performed": False,
        "optimizer_created": False,
        "f2_f3_performed": False,
        "test_split_used": False,
        "human_hidden_group_used": False,
        "forbidden_claims": [
            "F0/F1 does not establish gradient-signal validity.",
            "F0/F1 does not establish selection, SFT, or LoRA effectiveness.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen model-aware F0 static estimate and F1 two-sample memory smoke."
    )
    parser.add_argument(
        "--phase", choices=("f0", "f1", "all", "correct-audit"), default="all"
    )
    parser.add_argument("--output-dir", default="results/model_aware_signal_f0_f1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    prepare_output_dir(output_dir, args.phase)
    static_path = output_dir / "static_estimate.json"
    memory_path = output_dir / "memory_smoke.json"
    correct_audit_path = output_dir / "correct_query_length_audit.json"
    metadata_path = output_dir / "run_metadata.json"
    metadata = run_metadata_base(args.phase)

    if args.phase == "correct-audit":
        static_artifact = json.loads(static_path.read_text(encoding="utf-8"))
        correct_audit = run_correct_query_length_audit(static_artifact, static_path)
        write_json_atomic(correct_audit_path, correct_audit)
        if correct_audit["status"] != "passed":
            raise SystemExit(2)
        return

    if args.phase in {"all", "f0"}:
        static_artifact, candidate_row, error_row = run_f0()
        write_json_atomic(static_path, static_artifact)
        metadata["f0_status"] = static_artifact["status"]
    else:
        static_artifact = json.loads(static_path.read_text(encoding="utf-8"))
        candidates, errors, _ = load_frozen_rows()
        candidate_row = find_row_by_hash(
            candidates, static_artifact["longest_candidate"]["content_sha256"]
        )
        error_row = find_row_by_hash(
            errors, static_artifact["longest_error_query"]["content_sha256"]
        )
        metadata["f0_status"] = static_artifact["status"]

    exit_code = 0
    if args.phase in {"all", "f1"}:
        try:
            memory_artifact, f1_details = run_f1(static_artifact, candidate_row, error_row)
        except Exception as exc:
            memory_artifact = {
                "schema_version": 1,
                "stage_id": "model_aware_signal_f0_f1",
                "phase": "f1_memory_smoke",
                "status": "stopped",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": MODEL,
                "revision": REVISION,
                "samples": [],
                "hard_stop_reasons": [f"unexpected_error:{type(exc).__name__}:{exc}"],
                "claim_boundary": "F1 failed without fallback; no research score was produced.",
            }
            f1_details = {"status": "stopped", "packages": {}}
        write_json_atomic(memory_path, memory_artifact)
        metadata["f1_status"] = memory_artifact["status"]
        metadata["f1_details"] = f1_details
        if memory_artifact["status"] != "passed":
            exit_code = 2
    metadata["status"] = (
        "passed"
        if metadata.get("f0_status") == "passed"
        and (args.phase == "f0" or metadata.get("f1_status") == "passed")
        else "stopped"
    )
    write_json_atomic(metadata_path, metadata)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
