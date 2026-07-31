"""Pure, deterministic checks used by the formal B=500 audit command."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerFast

from eg_sft.experiment.formal_runtime import file_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while rejecting blank lines and non-object rows."""

    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"JSONL is missing its final newline: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one audit artifact without ever replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_sha256_sidecar_exclusive(path: Path) -> Path:
    """Bind a newly created artifact to an exclusive SHA-256 sidecar."""

    sidecar = path.with_name(path.name + ".sha256")
    digest = file_sha256(path)
    with sidecar.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{digest}  {path.name}\n")
    return sidecar


def load_tokenizer_snapshot(directory: Path) -> PreTrainedTokenizerFast:
    """Load tokenizer files locally without invoking Hub path resolution."""

    config = read_json(directory / "tokenizer_config.json")

    def token_content(value: Any) -> Any:
        if isinstance(value, Mapping):
            return value.get("content")
        return value

    additional = [token_content(value) for value in config.get("additional_special_tokens", [])]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(directory / "tokenizer.json"),
        bos_token=token_content(config.get("bos_token")),
        eos_token=token_content(config.get("eos_token")),
        pad_token=token_content(config.get("pad_token")),
        unk_token=token_content(config.get("unk_token")),
        additional_special_tokens=additional,
        clean_up_tokenization_spaces=bool(config.get("clean_up_tokenization_spaces", False)),
        model_max_length=int(config.get("model_max_length", 1_000_000_000)),
    )
    return tokenizer


def tokenizer_file_hashes(directory: Path) -> dict[str, str]:
    """Hash the tokenizer files that determine encoding and special tokens."""

    required = ("tokenizer.json", "tokenizer_config.json")
    result: dict[str, str] = {}
    for name in required:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"missing tokenizer file: {path}")
        result[name] = file_sha256(path)
    for name in ("vocab.json", "merges.txt", "special_tokens_map.json"):
        path = directory / name
        if path.is_file():
            result[name] = file_sha256(path)
    return result


def compare_tokenizer_texts(
    *,
    reference: Any,
    saved: Any,
    texts: Sequence[str],
    tokenizer_kwargs: Mapping[str, Any],
    single_item_batch: bool,
) -> dict[str, Any]:
    """Require exact token-ID equality for an ordered collection of texts."""

    mismatches: list[dict[str, int]] = []
    for index, text in enumerate(texts):
        if single_item_batch:
            reference_ids = reference([text], **tokenizer_kwargs)["input_ids"][0]
            saved_ids = saved([text], **tokenizer_kwargs)["input_ids"][0]
        else:
            reference_ids = reference(text, **tokenizer_kwargs)["input_ids"]
            saved_ids = saved(text, **tokenizer_kwargs)["input_ids"]
        if reference_ids != saved_ids:
            mismatches.append(
                {
                    "index": index,
                    "reference_length": len(reference_ids),
                    "saved_length": len(saved_ids),
                }
            )
    if mismatches:
        raise ValueError(
            f"tokenizer ID mismatch for {len(mismatches)} of {len(texts)} texts; "
            f"first={mismatches[0]}"
        )
    return {
        "compared_count": len(texts),
        "token_id_mismatch_count": 0,
        "exact_token_id_equality": True,
    }


def audit_training_contract(
    *,
    metrics: Mapping[str, Any],
    token_audit: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    training_config: Mapping[str, Any],
    strategy: str,
    seed: int,
    adapter_parameter_count: int,
) -> dict[str, Any]:
    """Check the frozen sample, optimizer, token and reload accounting."""

    if len(token_audit) != len(selected):
        raise ValueError("training token audit count differs from selected count")
    compared_fields = (
        "candidate_id",
        "source_index",
        "prompt_sha256",
        "response_sha256",
        "total_tokens",
        "supervised_tokens",
    )
    for index, (observed, expected) in enumerate(zip(token_audit, selected, strict=True)):
        for field in compared_fields:
            if observed.get(field) != expected.get(field):
                raise ValueError(f"training token audit mismatch at {index}: {field}")

    epochs = int(training_config["epochs"])
    accumulation = int(training_config["gradient_accumulation_steps"])
    total_micro_batches = len(selected) * epochs
    expected_steps = math.ceil(total_micro_batches / accumulation)
    expected_supervised_tokens = sum(int(row["supervised_tokens"]) for row in selected) * epochs
    expected_values = {
        "status": "PASS",
        "strategy": strategy,
        "seed": seed,
        "selected_count": len(selected),
        "epochs": epochs,
        "optimizer_steps_planned": expected_steps,
        "optimizer_steps_completed": expected_steps,
        "supervised_tokens_seen": expected_supervised_tokens,
        "trainable_parameters": adapter_parameter_count,
    }
    for key, expected in expected_values.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"training metric mismatch for {key}: {metrics.get(key)!r} != {expected!r}"
            )
    difference = float(metrics["adapter_reload_loss_absolute_difference"])
    if not math.isfinite(difference) or difference > 1e-6:
        raise ValueError("adapter reload loss difference exceeds 1e-6")
    if metrics.get("adapter_reload_gate_difference_at_most_1e_6") is not True:
        raise ValueError("adapter reload gate is not true")
    return {
        "selected_count": len(selected),
        "epochs": epochs,
        "optimizer_steps": expected_steps,
        "supervised_tokens_seen": expected_supervised_tokens,
        "token_audit_matches_selection": True,
        "adapter_reload_loss_absolute_difference": difference,
        "adapter_reload_gate_passed": True,
    }


def audit_checkpoint_directory(
    *,
    checkpoint_directory: Path,
    expected_binding: Mapping[str, Any],
    total_micro_batches: int,
    gradient_accumulation_steps: int,
    optimizer_steps: int,
    verify_payload_hashes: bool,
) -> dict[str, Any]:
    """Validate every immutable checkpoint/sidecar pair and its progress."""

    sidecar_paths = sorted(checkpoint_directory.glob("checkpoint_*.json"))
    payload_paths = sorted(checkpoint_directory.glob("checkpoint_*.pt"))
    expected_count = optimizer_steps + 1
    if len(sidecar_paths) != expected_count or len(payload_paths) != expected_count:
        raise ValueError(
            "checkpoint count mismatch: "
            f"sidecars={len(sidecar_paths)}, payloads={len(payload_paths)}, "
            f"expected={expected_count}"
        )
    expected_progress = [(0, 0)] + [
        (min(step * gradient_accumulation_steps, total_micro_batches), step)
        for step in range(1, optimizer_steps + 1)
    ]
    observed_progress: list[tuple[int, int]] = []
    mapped_payloads: list[str] = []
    for sidecar_path in sidecar_paths:
        sidecar = read_json(sidecar_path)
        for key, value in expected_binding.items():
            if sidecar.get(key) != value:
                raise ValueError(f"checkpoint binding mismatch at {sidecar_path.name}: {key}")
        digest = sidecar.get("checkpoint_sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid checkpoint hash in {sidecar_path.name}")
        payload_name = sidecar.get("checkpoint_file")
        if not isinstance(payload_name, str):
            raise ValueError(f"missing checkpoint filename in {sidecar_path.name}")
        payload_path = checkpoint_directory / payload_name
        if not payload_path.is_file():
            raise FileNotFoundError(f"missing checkpoint payload: {payload_path}")
        if verify_payload_hashes and file_sha256(payload_path) != digest:
            raise ValueError(f"checkpoint payload hash mismatch: {payload_name}")
        mapped_payloads.append(payload_name)
        observed_progress.append(
            (
                int(sidecar["next_micro_batch_index"]),
                int(sidecar["optimizer_steps"]),
            )
        )
    if len(set(mapped_payloads)) != len(mapped_payloads):
        raise ValueError("multiple sidecars map to the same checkpoint payload")
    if sorted(mapped_payloads) != [path.name for path in payload_paths]:
        raise ValueError("checkpoint sidecars do not map exactly onto payload files")
    if observed_progress != expected_progress:
        raise ValueError("checkpoint progress sequence differs from the frozen schedule")
    return {
        "sidecar_count": len(sidecar_paths),
        "payload_count": len(payload_paths),
        "first_progress": list(observed_progress[0]),
        "last_progress": list(observed_progress[-1]),
        "all_bindings_match": True,
        "all_payload_hashes_verified": verify_payload_hashes,
        "unique_sidecar_payload_mapping": True,
    }


def audit_thermal_events(
    *,
    events: Sequence[Mapping[str, Any]],
    pause_at_c: float,
    resume_at_c: float,
    hard_stop_at_c: float,
) -> dict[str, Any]:
    """Check that every recorded pause obeys the frozen thermal policy."""

    initial_temperatures: list[float] = []
    resume_temperatures: list[float] = []
    stages: Counter[str] = Counter()
    for index, event in enumerate(events):
        if event.get("event") != "thermal_pause":
            raise ValueError(f"unexpected thermal event at row {index}")
        initial = float(event["initial_sample"]["temperature_c"])
        resumed = float(event["resume_sample"]["temperature_c"])
        if initial < pause_at_c:
            raise ValueError(f"thermal pause below threshold at row {index}")
        if initial >= hard_stop_at_c:
            raise ValueError(f"thermal pause reached hard-stop threshold at row {index}")
        if resumed > resume_at_c:
            raise ValueError(f"thermal resume above threshold at row {index}")
        initial_temperatures.append(initial)
        resume_temperatures.append(resumed)
        stages[str(event["stage"])] += 1
    return {
        "event_count": len(events),
        "stage_counts": dict(sorted(stages.items())),
        "max_pause_temperature_c": max(initial_temperatures, default=None),
        "max_resume_temperature_c": max(resume_temperatures, default=None),
        "hard_stop_event_count": 0,
        "all_events_follow_policy": True,
    }


def audit_formal_output_scope(
    *,
    actual_jobs: Sequence[tuple[str, int]],
    job_order: Sequence[Mapping[str, Any]],
    current_job: tuple[str, int],
) -> dict[str, Any]:
    """Require formal output directories to equal the schedule prefix."""

    schedule = [(str(job["strategy"]), int(job["seed"])) for job in job_order]
    if current_job not in schedule:
        raise ValueError(f"current job is absent from frozen schedule: {current_job}")
    prefix = schedule[: schedule.index(current_job) + 1]
    if len(set(actual_jobs)) != len(actual_jobs):
        raise ValueError("duplicate formal job directories detected")
    if set(actual_jobs) != set(prefix) or len(actual_jobs) != len(prefix):
        raise ValueError(f"formal output scope mismatch: actual={actual_jobs}, expected={prefix}")
    return {
        "completed_job_count": len(actual_jobs),
        "completed_jobs": [
            {"strategy": strategy, "seed": seed} for strategy, seed in actual_jobs
        ],
        "matches_frozen_schedule_prefix": True,
        "next_jobs_not_started": True,
    }
