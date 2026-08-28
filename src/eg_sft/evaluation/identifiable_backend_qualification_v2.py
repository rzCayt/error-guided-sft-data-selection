"""Auditable state machine for batched Transformers qualification v2.

The module is CPU-only and imports no CUDA framework.  The GPU runner imports
training dependencies only after its ``--contract-only`` exit point.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "identifiable-transformers-qualification-v2"
VALIDATED_CONFIG_SHA_KEY = "_validated_config_sha256"
SEMANTIC_FIELDS = (
    "record_id",
    "parsed_prediction",
    "numeric_correct",
    "strict_parse_status",
    "parse_mode",
    "parse_status",
)
MODEL_IDS = (
    "base",
    "random_common_rep1_seed17",
    "rds_error_common_rep1_seed17",
)
BATCH_SIZES = (1, 2, 4, 8)
RESUME_PROBE = {
    "stage": "smoke128",
    "model_id": "base",
    "batch_size": 2,
    "stop_after_records": 64,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_exclusive_or_verify(path: Path, payload: bytes) -> str:
    """Create an immutable file, or verify an existing byte-identical file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable artifact changed: {path.name}")
        return "VERIFIED_IDENTICAL"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return "CREATED"


def append_jsonl_fsynced(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_config(*, repo_root: Path, config_path: Path) -> dict[str, Any]:
    resolved_config_path = config_path.resolve()
    config = read_json(resolved_config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected qualification config version")
    matrix = config.get("matrix", {})
    matrix_path = (repo_root / str(matrix.get("path", ""))).resolve()
    if not matrix_path.is_file() or file_sha256(matrix_path) != matrix.get("sha256"):
        raise ValueError("identifiable v4 matrix SHA-256 changed")
    if [row.get("adapter_id") for row in config.get("adapters", [])] != list(
        MODEL_IDS[1:]
    ):
        raise ValueError("qualification must bind exactly the two audited adapters")
    stages = config.get("stages", {})
    if stages.get("smoke128", {}).get("batch_sizes") != list(BATCH_SIZES):
        raise ValueError("smoke batch-size grid changed")
    if stages.get("smoke128", {}).get("record_count") != 128:
        raise ValueError("smoke record count changed")
    if stages.get("confirm512", {}).get("record_count") != 512:
        raise ValueError("confirm record count changed")
    if stages.get("shadow3841", {}).get("record_count") != 3841:
        raise ValueError("shadow record count changed")
    if config.get("resume_probe") != RESUME_PROBE:
        raise ValueError("resume probe must remain smoke128/base/batch2 at 64 records")
    gates = config.get("gates", {})
    if tuple(gates.get("semantic_fields", [])) != SEMANTIC_FIELDS:
        raise ValueError("semantic equivalence fields changed")
    if float(gates.get("minimum_examples_per_second_speedup", 0)) != 1.5:
        raise ValueError("throughput threshold changed")
    if float(gates.get("minimum_shadow_wall_time_reduction", 0)) != 0.25:
        raise ValueError("wall-time threshold changed")
    config[VALIDATED_CONFIG_SHA_KEY] = file_sha256(resolved_config_path)
    return config


def validate_stop_after_records_request(
    *,
    stage: str | None,
    model_id: str | None,
    batch_size: int | None,
    stop_after_records: int | None,
    config: Mapping[str, Any],
) -> None:
    """Allow an intentional early stop only for the frozen resume probe.

    The follow-up invocation that completes the same run omits
    ``stop_after_records``. All other runs must be uninterrupted so their
    throughput and wall-clock measurements represent one clean attempt.
    """

    if stop_after_records is None:
        return
    probe = config["resume_probe"]
    observed = {
        "stage": stage,
        "model_id": model_id,
        "batch_size": batch_size,
        "stop_after_records": stop_after_records,
    }
    if observed != probe:
        raise ValueError(
            "--stop-after-records is reserved for "
            "smoke128/base/batch2 with value 64"
        )


def adapter_binding(config: Mapping[str, Any], adapter_id: str) -> dict[str, Any]:
    matches = [row for row in config["adapters"] if row["adapter_id"] == adapter_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicated adapter ID: {adapter_id}")
    return dict(matches[0])


def validate_adapter_root(
    *, adapter_root: Path, binding: Mapping[str, Any]
) -> dict[str, str]:
    """Require formal audit, training metrics and safetensors to agree."""

    root = adapter_root.resolve()
    formal_path = root / "audit" / "formal_cell_audit.json"
    training_path = root / "training_complete" / "training_metrics.json"
    adapter_path = root / "training_complete" / "adapter" / "adapter_model.safetensors"
    for path in (formal_path, training_path, adapter_path):
        if not path.is_file():
            raise ValueError(f"adapter evidence is missing: {path.name}")
    hashes = {
        "formal_audit_sha256": file_sha256(formal_path),
        "training_metrics_sha256": file_sha256(training_path),
        "adapter_model_sha256": file_sha256(adapter_path),
    }
    for field, actual in hashes.items():
        if actual != binding.get(field):
            raise ValueError(f"adapter {field} changed")
    formal = read_json(formal_path)
    training = read_json(training_path)
    if formal.get("status") != "PASS" or formal.get("accuracy_withheld") is not True:
        raise ValueError("formal adapter audit is not a blind PASS")
    if training.get("status") != "PASS":
        raise ValueError("adapter training metrics are not PASS")
    triple = {
        str(formal.get("artifact_hashes", {}).get("adapter_model")),
        str(training.get("adapter_model_sha256")),
        hashes["adapter_model_sha256"],
    }
    if triple != {str(binding["adapter_model_sha256"])}:
        raise ValueError("formal/training/safetensors adapter SHA binding disagrees")
    return hashes


def validate_prefix(
    *, rows: Sequence[Mapping[str, Any]], frozen_ids: Sequence[str]
) -> int:
    if len(rows) > len(frozen_ids):
        raise ValueError("output is longer than the frozen records")
    observed = [str(row.get("record_id")) for row in rows]
    if observed != list(frozen_ids[: len(observed)]):
        raise ValueError("output is not a complete ordered prefix")
    if len(observed) != len(set(observed)):
        raise ValueError("output contains duplicate record IDs")
    return len(observed)


def compare_rows(
    *,
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    record_count: int,
) -> dict[str, Any]:
    if len(reference) != record_count or len(candidate) != record_count:
        return {"status": "FAIL", "reason": "record_count_mismatch"}
    semantic_mismatches: list[dict[str, Any]] = []
    token_mismatches: list[str] = []
    for left, right in zip(reference, candidate, strict=True):
        differences = {
            field: {"reference": left.get(field), "candidate": right.get(field)}
            for field in SEMANTIC_FIELDS
            if field not in left or field not in right or left.get(field) != right.get(field)
        }
        if differences:
            semantic_mismatches.append(
                {"record_id": left.get("record_id"), "differences": differences}
            )
        if (
            "generated_token_ids" not in left
            or "generated_token_ids" not in right
            or list(left.get("generated_token_ids", []))
            != list(right.get("generated_token_ids", []))
        ):
            token_mismatches.append(str(left.get("record_id")))
    semantic_equal = not semantic_mismatches
    token_equal = not token_mismatches
    token_gate = token_equal if record_count in {128, 512} else (
        token_equal or (record_count == 3841 and semantic_equal)
    )
    return {
        "status": "PASS" if semantic_equal and token_gate else "FAIL",
        "record_count": record_count,
        "semantic_equal": semantic_equal,
        "token_ids_equal": token_equal,
        "semantic_fallback_used": bool(
            record_count == 3841 and semantic_equal and not token_equal
        ),
        "semantic_mismatch_count": len(semantic_mismatches),
        "token_mismatch_count": len(token_mismatches),
        "semantic_mismatches": semantic_mismatches[:20],
        "token_mismatch_record_ids": token_mismatches[:20],
    }


def run_key(*, stage: str, model_id: str, batch_size: int) -> str:
    return f"{stage}__{model_id}__b{batch_size}"


def run_dir(
    *, root: Path, stage: str, model_id: str, batch_size: int
) -> Path:
    return root.resolve() / "runs" / run_key(
        stage=stage, model_id=model_id, batch_size=batch_size
    )


def next_attempt_id(attempts_path: Path) -> int:
    starts = [row for row in read_jsonl(attempts_path) if row.get("event") == "START"]
    return len(starts) + 1


def record_attempt_start(
    *, run_directory: Path, start_index: int, stop_after_records: int | None
) -> int:
    attempt_id = next_attempt_id(run_directory / "runtime_attempts.jsonl")
    append_jsonl_fsynced(
        run_directory / "runtime_attempts.jsonl",
        {
            "event": "START",
            "attempt_id": attempt_id,
            "start_index": start_index,
            "stop_after_records": stop_after_records,
            "resume_observed": start_index > 0,
        },
    )
    return attempt_id


def record_attempt_finish(
    *,
    run_directory: Path,
    attempt_id: int,
    start_index: int,
    end_index: int,
    generation_seconds: float,
    full_wall_seconds: float,
    generated_tokens: int,
    peak_memory_bytes: int,
    stopped_early: bool,
    attempt_status: str = "PASS",
) -> None:
    append_jsonl_fsynced(
        run_directory / "runtime_attempts.jsonl",
        {
            "event": "FINISH",
            "attempt_id": attempt_id,
            "start_index": start_index,
            "end_index": end_index,
            "generated_records": end_index - start_index,
            "generation_seconds": generation_seconds,
            "full_wall_seconds": full_wall_seconds,
            "generated_tokens": generated_tokens,
            "peak_memory_bytes": peak_memory_bytes,
            "resume_observed": start_index > 0,
            "stopped_early": stopped_early,
            "attempt_status": attempt_status,
        },
    )


def aggregate_attempts(run_directory: Path) -> dict[str, Any]:
    finished = [
        row
        for row in read_jsonl(run_directory / "runtime_attempts.jsonl")
        if row.get("event") == "FINISH"
    ]
    if not finished:
        raise ValueError("run has no finished generation attempt")
    return {
        "attempt_count": len(finished),
        "generation_seconds": sum(float(row["generation_seconds"]) for row in finished),
        "full_wall_seconds": sum(float(row["full_wall_seconds"]) for row in finished),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in finished),
        "generated_records": sum(int(row["generated_records"]) for row in finished),
        "peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in finished),
        "resume_observed": any(bool(row.get("resume_observed")) for row in finished),
        "failed_attempt_count": sum(
            str(row.get("attempt_status", "PASS")) == "FAIL" for row in finished
        ),
    }


def finalize_run(
    *,
    run_directory: Path,
    frozen_ids: Sequence[str],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows_path = run_directory / "raw_outputs.jsonl"
    rows = read_jsonl(rows_path)
    if validate_prefix(rows=rows, frozen_ids=frozen_ids) != len(frozen_ids):
        raise ValueError("cannot write metrics for an incomplete output")
    attempts = aggregate_attempts(run_directory)
    if attempts["generated_records"] != len(rows):
        raise ValueError("attempt record accounting changed")
    metrics = {
        "status": "PASS",
        "record_count": len(rows),
        "rows_sha256": file_sha256(rows_path),
        **attempts,
        "examples_per_second": len(rows) / attempts["generation_seconds"],
        "accuracy_withheld": True,
    }
    for field in (
        "base_model_repo_id",
        "base_model_revision",
        "tokenizer_repo_id",
        "tokenizer_revision",
        "formal_audit_sha256",
        "training_metrics_sha256",
        "adapter_model_sha256",
        "config_sha256",
        "matrix_sha256",
        "script_sha256",
        "module_sha256",
    ):
        metrics[field] = manifest.get(field)
    final_manifest = {
        **dict(manifest),
        "rows_sha256": metrics["rows_sha256"],
        "peak_memory_bytes": metrics["peak_memory_bytes"],
        "examples_per_second": metrics["examples_per_second"],
        "generation_seconds": metrics["generation_seconds"],
        "full_wall_seconds": metrics["full_wall_seconds"],
    }
    write_exclusive_or_verify(
        run_directory / "metrics.json", canonical_json_bytes(metrics)
    )
    write_exclusive_or_verify(
        run_directory / "manifest.json", canonical_json_bytes(final_manifest)
    )
    return metrics


def write_replay_probe(run_directory: Path) -> Path:
    """Verify completed bytes and persist a separate immutable replay probe."""

    metrics_path = run_directory / "metrics.json"
    manifest_path = run_directory / "manifest.json"
    rows_path = run_directory / "raw_outputs.jsonl"
    if not all(path.is_file() for path in (metrics_path, manifest_path, rows_path)):
        raise ValueError("replay probe requires a completed run")
    metrics = read_json(metrics_path)
    manifest = read_json(manifest_path)
    if metrics.get("rows_sha256") != file_sha256(rows_path):
        raise ValueError("completed rows changed before replay")
    if manifest.get("rows_sha256") != metrics.get("rows_sha256"):
        raise ValueError("manifest/metrics row hash changed")
    probe_dir = run_directory / "replay_probes"
    existing = sorted(probe_dir.glob("replay_probe_*.json")) if probe_dir.exists() else []
    path = probe_dir / f"replay_probe_{len(existing) + 1:04d}.json"
    payload = {
        "status": "PASS",
        "rows_sha256": file_sha256(rows_path),
        "metrics_sha256": file_sha256(metrics_path),
        "manifest_sha256": file_sha256(manifest_path),
        "bytes_verified_identical": True,
    }
    write_exclusive_or_verify(path, canonical_json_bytes(payload))
    return path


def write_failure_artifact(
    *, root: Path, stage: str, model_id: str, batch_size: int, error: BaseException
) -> Path:
    path = root.resolve() / "failures" / f"{run_key(stage=stage, model_id=model_id, batch_size=batch_size)}.json"
    payload = {
        "status": "INELIGIBLE" if batch_size > 1 else "FATAL",
        "stage": stage,
        "model_id": model_id,
        "batch_size": batch_size,
        "error_type": type(error).__name__,
        "error_message": str(error)[:1000],
    }
    write_exclusive_or_verify(path, canonical_json_bytes(payload))
    return path


def _completed_run(root: Path, stage: str, model_id: str, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = run_dir(root=root, stage=stage, model_id=model_id, batch_size=batch_size)
    return read_jsonl(directory / "raw_outputs.jsonl"), read_json(directory / "metrics.json")


def select_smoke_best(*, root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Select one unique batch by mean throughput across all three models."""

    root = root.resolve()
    fatal_batch1 = [
        model_id
        for model_id in MODEL_IDS
        if (
            root
            / "failures"
            / f"{run_key(stage='smoke128', model_id=model_id, batch_size=1)}.json"
        ).is_file()
    ]
    if fatal_batch1:
        payload = {
            "status": "FAIL",
            "reason": "batch1_failure_is_fatal",
            "models": fatal_batch1,
            "matrix_sha256": config["matrix"]["sha256"],
        }
        write_exclusive_or_verify(
            root / "smoke_selection.json", canonical_json_bytes(payload)
        )
        return payload
    for model_id in MODEL_IDS:
        directory = run_dir(root=root, stage="smoke128", model_id=model_id, batch_size=1)
        if not (directory / "metrics.json").is_file():
            raise ValueError(f"batch1 smoke failed or is incomplete: {model_id}")
    candidates: list[dict[str, Any]] = []
    for batch_size in (2, 4, 8):
        model_rows = []
        reasons = []
        for model_id in MODEL_IDS:
            directory = run_dir(
                root=root, stage="smoke128", model_id=model_id, batch_size=batch_size
            )
            if not (directory / "metrics.json").is_file():
                reasons.append(f"{model_id}:missing_or_failed")
                continue
            reference, _ = _completed_run(root, "smoke128", model_id, 1)
            candidate, metrics = _completed_run(
                root, "smoke128", model_id, batch_size
            )
            comparison = compare_rows(
                reference=reference, candidate=candidate, record_count=128
            )
            if comparison["status"] != "PASS":
                reasons.append(f"{model_id}:equivalence_failed")
            else:
                model_rows.append(
                    {
                        "model_id": model_id,
                        "examples_per_second": float(metrics["examples_per_second"]),
                        "comparison": comparison,
                        "evidence": {
                            "reference_rows_sha256": file_sha256(
                                run_dir(
                                    root=root,
                                    stage="smoke128",
                                    model_id=model_id,
                                    batch_size=1,
                                )
                                / "raw_outputs.jsonl"
                            ),
                            "candidate_rows_sha256": file_sha256(
                                directory / "raw_outputs.jsonl"
                            ),
                            "candidate_metrics_sha256": file_sha256(
                                directory / "metrics.json"
                            ),
                        },
                    }
                )
        eligible = len(model_rows) == len(MODEL_IDS) and not reasons
        candidates.append(
            {
                "batch_size": batch_size,
                "eligible": eligible,
                "mean_examples_per_second": (
                    mean(row["examples_per_second"] for row in model_rows)
                    if eligible
                    else None
                ),
                "models": model_rows,
                "reasons": reasons,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        payload = {"status": "FAIL", "reason": "no_eligible_batched_candidate", "candidates": candidates}
    else:
        ranked = sorted(
            eligible,
            key=lambda row: (-float(row["mean_examples_per_second"]), int(row["batch_size"])),
        )
        if len(ranked) > 1 and ranked[0]["mean_examples_per_second"] == ranked[1]["mean_examples_per_second"]:
            payload = {"status": "FAIL", "reason": "best_batch_not_unique", "candidates": candidates}
        else:
            payload = {
                "status": "PASS",
                "best_batch_size": ranked[0]["batch_size"],
                "selection_rule": "unique maximum mean examples/s across base plus two audited adapters",
                "matrix_sha256": config["matrix"]["sha256"],
                "candidates": candidates,
            }
    output = root / "smoke_selection.json"
    write_exclusive_or_verify(output, canonical_json_bytes(payload))
    return payload


def validate_run_manifest(
    *, manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    required = (
        "stage",
        "model_id",
        "batch_size",
        "record_count",
        "base_model_repo_id",
        "base_model_revision",
        "tokenizer_repo_id",
        "tokenizer_revision",
        "config_sha256",
        "matrix_sha256",
        "script_sha256",
        "module_sha256",
        "rows_sha256",
        "peak_memory_bytes",
        "examples_per_second",
        "generation_seconds",
        "full_wall_seconds",
    )
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(f"run manifest is missing provenance: {missing}")
    expected_config_sha256 = config.get(VALIDATED_CONFIG_SHA_KEY)
    if not isinstance(expected_config_sha256, str):
        raise ValueError("validated qualification config SHA is unavailable")
    if manifest["config_sha256"] != expected_config_sha256:
        raise ValueError("run manifest qualification config SHA changed")
    if manifest["matrix_sha256"] != config["matrix"]["sha256"]:
        raise ValueError("run manifest matrix SHA changed")
    expected_bindings = {
        "base_model_repo_id": config["base_model"]["repo_id"],
        "base_model_revision": config["base_model"]["revision"],
        "tokenizer_repo_id": config["tokenizer"]["repo_id"],
        "tokenizer_revision": config["tokenizer"]["revision"],
    }
    for field, expected in expected_bindings.items():
        if manifest.get(field) != expected:
            raise ValueError(f"run manifest {field} changed")
    if manifest.get("model_id") != "base":
        adapter = adapter_binding(config, str(manifest.get("model_id")))
        for field in (
            "formal_audit_sha256",
            "training_metrics_sha256",
            "adapter_model_sha256",
        ):
            if manifest.get(field) != adapter[field]:
                raise ValueError(f"run manifest adapter {field} changed")


def _require_replay(directory: Path) -> bool:
    rows_path = directory / "raw_outputs.jsonl"
    metrics_path = directory / "metrics.json"
    manifest_path = directory / "manifest.json"
    if not all(path.is_file() for path in (rows_path, metrics_path, manifest_path)):
        return False
    probes = (
        sorted((directory / "replay_probes").glob("replay_probe_*.json"))
        if (directory / "replay_probes").is_dir()
        else []
    )
    expected = {
        "rows_sha256": file_sha256(rows_path),
        "metrics_sha256": file_sha256(metrics_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    return bool(probes) and all(
        read_json(path).get("status") == "PASS"
        and read_json(path).get("bytes_verified_identical") is True
        and all(read_json(path).get(field) == value for field, value in expected.items())
        for path in probes
    )


def audit_smoke128(
    *, root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Close all 128-only gates without reading 512/3,841 artifacts.

    This is deliberately a prequalification artifact.  It never authorizes the
    formal batched backend used by the later matrix, because the frozen full
    qualification still requires confirm512 and shadow3841.
    """

    root = root.resolve()
    selection = select_smoke_best(root=root, config=config)
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    explicit_ineligible: list[dict[str, Any]] = []
    unaccounted: list[str] = []
    provenance_errors: list[dict[str, str]] = []
    replay_failures: list[str] = []
    attempt_failures: list[str] = []
    accuracy_failures: list[str] = []

    for model_id in MODEL_IDS:
        for batch_size in BATCH_SIZES:
            key = run_key(
                stage="smoke128", model_id=model_id, batch_size=batch_size
            )
            directory = run_dir(
                root=root,
                stage="smoke128",
                model_id=model_id,
                batch_size=batch_size,
            )
            metrics_path = directory / "metrics.json"
            manifest_path = directory / "manifest.json"
            rows_path = directory / "raw_outputs.jsonl"
            if metrics_path.is_file() and manifest_path.is_file() and rows_path.is_file():
                metrics = read_json(metrics_path)
                manifest = read_json(manifest_path)
                errors: list[str] = []
                try:
                    validate_run_manifest(manifest=manifest, config=config)
                except ValueError as error:
                    errors.append(str(error))
                if manifest.get("stage") != "smoke128":
                    errors.append("manifest stage changed")
                if manifest.get("model_id") != model_id:
                    errors.append("manifest model ID changed")
                if int(manifest.get("batch_size", -1)) != batch_size:
                    errors.append("manifest batch size changed")
                if int(manifest.get("record_count", -1)) != 128:
                    errors.append("manifest record count changed")
                actual_rows_sha256 = file_sha256(rows_path)
                if metrics.get("rows_sha256") != actual_rows_sha256:
                    errors.append("metrics row SHA changed")
                if manifest.get("rows_sha256") != actual_rows_sha256:
                    errors.append("manifest row SHA changed")
                if int(metrics.get("record_count", -1)) != 128:
                    errors.append("metrics record count changed")
                if errors:
                    provenance_errors.append({"run_key": key, "error": "; ".join(errors)})
                if not _require_replay(directory):
                    replay_failures.append(key)
                is_probe = model_id == "base" and batch_size == 2
                if is_probe:
                    attempt_ok = (
                        metrics.get("resume_observed") is True
                        and int(metrics.get("attempt_count", 0)) == 2
                        and int(metrics.get("generated_records", 0)) == 128
                        and int(metrics.get("failed_attempt_count", 0)) == 0
                    )
                else:
                    attempt_ok = (
                        metrics.get("resume_observed") is False
                        and int(metrics.get("attempt_count", 0)) == 1
                        and int(metrics.get("generated_records", 0)) == 128
                        and int(metrics.get("failed_attempt_count", 0)) == 0
                    )
                if not attempt_ok:
                    attempt_failures.append(key)
                if (
                    metrics.get("accuracy_withheld") is not True
                    or manifest.get("accuracy_withheld") is not True
                ):
                    accuracy_failures.append(key)
                completed[(model_id, batch_size)] = {
                    "run_key": key,
                    "directory": directory,
                    "metrics": metrics,
                }
                continue

            failure_path = root / "failures" / f"{key}.json"
            if batch_size > 1 and failure_path.is_file():
                failure = read_json(failure_path)
                if (
                    failure.get("status") == "INELIGIBLE"
                    and failure.get("stage") == "smoke128"
                    and failure.get("model_id") == model_id
                    and int(failure.get("batch_size", -1)) == batch_size
                ):
                    explicit_ineligible.append(
                        {"run_key": key, "failure_sha256": file_sha256(failure_path)}
                    )
                    continue
            unaccounted.append(key)

    comparisons: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        reference_entry = completed.get((model_id, 1))
        if reference_entry is None:
            continue
        reference_rows = read_jsonl(
            Path(reference_entry["directory"]) / "raw_outputs.jsonl"
        )
        for batch_size in BATCH_SIZES[1:]:
            candidate_entry = completed.get((model_id, batch_size))
            if candidate_entry is None:
                continue
            comparison = compare_rows(
                reference=reference_rows,
                candidate=read_jsonl(
                    Path(candidate_entry["directory"]) / "raw_outputs.jsonl"
                ),
                record_count=128,
            )
            comparisons.append(
                {"model_id": model_id, "batch_size": batch_size, **comparison}
            )

    selected_speedups: list[dict[str, Any]] = []
    if selection.get("status") == "PASS":
        selected_batch_size = int(selection["best_batch_size"])
        for model_id in MODEL_IDS:
            reference = completed.get((model_id, 1))
            candidate = completed.get((model_id, selected_batch_size))
            if reference is None or candidate is None:
                continue
            reference_speed = float(reference["metrics"]["examples_per_second"])
            candidate_speed = float(candidate["metrics"]["examples_per_second"])
            selected_speedups.append(
                {
                    "model_id": model_id,
                    "batch_size": selected_batch_size,
                    "speedup": candidate_speed / reference_speed,
                }
            )

    probe_key = ("base", 2)
    probe_passed = probe_key in completed and run_key(
        stage="smoke128", model_id="base", batch_size=2
    ) not in attempt_failures
    other_attempts_clean = all(
        entry["run_key"] not in attempt_failures
        for key, entry in completed.items()
        if key != probe_key
    )
    gates = {
        "all_expected_smoke_runs_accounted": not unaccounted,
        "batch1_complete_for_all_models": all(
            (model_id, 1) in completed for model_id in MODEL_IDS
        ),
        "all_completed_rows_and_manifests_bound": not provenance_errors,
        "all_completed_candidate_token_ids_equal": bool(comparisons)
        and all(row.get("status") == "PASS" for row in comparisons),
        "selected_batch_speedup_at_least_1_5x_for_each_model": (
            len(selected_speedups) == len(MODEL_IDS)
            and all(
                float(row["speedup"])
                >= float(config["gates"]["minimum_examples_per_second_speedup"])
                for row in selected_speedups
            )
        ),
        "fixed_smoke_resume_probe_passed": probe_passed,
        "other_completed_runs_single_attempt_without_resume": other_attempts_clean,
        "all_completed_outputs_replay_verified": not replay_failures
        and bool(completed),
        "accuracy_remained_withheld": not accuracy_failures and bool(completed),
    }
    report = {
        "schema_version": f"{SCHEMA_VERSION}-smoke128-audit-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "qualification_config_sha256": config[VALIDATED_CONFIG_SHA_KEY],
        "matrix_sha256": config["matrix"]["sha256"],
        "audit_module_sha256": file_sha256(Path(__file__)),
        "gates": gates,
        "selection": selection,
        "selected_speedups": selected_speedups,
        "comparisons": comparisons,
        "completed_run_count": len(completed),
        "explicit_ineligible_runs": explicit_ineligible,
        "unaccounted_runs": unaccounted,
        "provenance_errors": provenance_errors,
        "replay_failures": replay_failures,
        "attempt_failures": attempt_failures,
        "accuracy_failures": accuracy_failures,
        "accuracy_withheld": True,
        "audit_gpu_accessed": False,
        "source_gpu_artifacts": True,
        "formal_backend_authorized": False,
        "larger_stage_artifacts_read": False,
        "method_effectiveness_claimed": False,
        "next_action": "STOP_AND_REVIEW",
    }
    write_exclusive_or_verify(
        root / "qualification_smoke128.json", canonical_json_bytes(report)
    )
    return report


def performance_gates(
    *, speedups: Sequence[float], shadow_wall_reduction: float
) -> dict[str, bool]:
    if not speedups:
        raise ValueError("performance gates require measured speedups")
    return {
        "throughput": all(float(value) >= 1.5 for value in speedups),
        "shadow_wall": float(shadow_wall_reduction) >= 0.25,
    }


def audit_final(
    *, root: Path, config: Mapping[str, Any], shadow_adapter_id: str
) -> dict[str, Any]:
    if shadow_adapter_id not in MODEL_IDS[1:]:
        raise ValueError("shadow adapter must be one of the two audited adapters")
    selection = read_json(root.resolve() / "smoke_selection.json")
    if selection.get("status") != "PASS":
        raise ValueError("smoke selection has not passed")
    best = int(selection["best_batch_size"])
    expected = []
    for model_id in MODEL_IDS:
        for batch_size in (1, best):
            expected.append(("confirm512", model_id, batch_size, 512))
    for batch_size in (1, best):
        expected.append(("shadow3841", shadow_adapter_id, batch_size, 3841))

    comparisons = []
    speedups = []
    replay_all = True
    performance_runs_clean = True
    for stage, model_id, batch_size, record_count in expected:
        directory = run_dir(
            root=root, stage=stage, model_id=model_id, batch_size=batch_size
        )
        metrics = read_json(directory / "metrics.json")
        manifest = read_json(directory / "manifest.json")
        validate_run_manifest(manifest=manifest, config=config)
        if metrics.get("rows_sha256") != file_sha256(directory / "raw_outputs.jsonl"):
            raise ValueError("final audit row SHA changed")
        performance_runs_clean = performance_runs_clean and (
            metrics.get("resume_observed") is False
            and int(metrics.get("attempt_count", 0)) == 1
        )
        replay_all = replay_all and _require_replay(directory)
        if batch_size == best:
            reference, reference_metrics = _completed_run(root, stage, model_id, 1)
            candidate, candidate_metrics = _completed_run(root, stage, model_id, best)
            comparison = compare_rows(
                reference=reference,
                candidate=candidate,
                record_count=record_count,
            )
            comparisons.append({"stage": stage, "model_id": model_id, **comparison})
            speedups.append(
                {
                    "stage": stage,
                    "model_id": model_id,
                    "speedup": float(candidate_metrics["examples_per_second"])
                    / float(reference_metrics["examples_per_second"]),
                }
            )

    # Resume evidence is deliberately isolated from every performance-gate
    # run. Only the frozen smoke128/base/batch2 probe may stop at record 64
    # and resume. Confirm512 and shadow3841 must each be a single fresh
    # attempt so their throughput and wall time are not biased by recovery.
    probe = config["resume_probe"]
    probe_directory = run_dir(
        root=root,
        stage=str(probe["stage"]),
        model_id=str(probe["model_id"]),
        batch_size=int(probe["batch_size"]),
    )
    probe_metrics = read_json(probe_directory / "metrics.json")
    resume_probe_passed = (
        int(probe_metrics.get("record_count", 0)) == 128
        and probe_metrics.get("resume_observed") is True
        and int(probe_metrics.get("attempt_count", 0)) == 2
        and int(probe_metrics.get("generated_records", 0)) == 128
        and int(probe_metrics.get("failed_attempt_count", 0)) == 0
    )

    # Non-overwrite remains a run-level engineering claim and is checked
    # across every completed qualification run.
    completed_directories = [
        path.parent
        for path in sorted((root.resolve() / "runs").glob("*/metrics.json"))
    ]
    if not completed_directories:
        raise ValueError("final audit found no completed qualification runs")
    replay_all = all(_require_replay(directory) for directory in completed_directories)

    shadow_reference = read_json(
        run_dir(root=root, stage="shadow3841", model_id=shadow_adapter_id, batch_size=1)
        / "metrics.json"
    )
    shadow_candidate = read_json(
        run_dir(root=root, stage="shadow3841", model_id=shadow_adapter_id, batch_size=best)
        / "metrics.json"
    )
    wall_reduction = 1.0 - float(shadow_candidate["full_wall_seconds"]) / float(
        shadow_reference["full_wall_seconds"]
    )
    performance = performance_gates(
        speedups=[float(row["speedup"]) for row in speedups],
        shadow_wall_reduction=wall_reduction,
    )
    detailed_gates = {
        "all_equivalence_passed": all(row["status"] == "PASS" for row in comparisons),
        "all_512_and_shadow_speedups_at_least_1_5x": performance["throughput"],
        "shadow_full_wall_reduction_at_least_25_percent": performance[
            "shadow_wall"
        ],
        "fixed_smoke_resume_probe_passed": resume_probe_passed,
        "performance_runs_single_attempt_without_resume": performance_runs_clean,
        "actual_resume_observed": resume_probe_passed and performance_runs_clean,
        "all_successful_outputs_replay_verified": replay_all,
    }
    gates = {
        "row_level_equivalence": all(
            row.get("semantic_equal") is True for row in comparisons
        ),
        "token_ids_equal_or_full_shadow_semantic_equivalence": all(
            row["status"] == "PASS" for row in comparisons
        ),
        "throughput_at_least_1_5x": detailed_gates[
            "all_512_and_shadow_speedups_at_least_1_5x"
        ],
        "full_cell_wall_time_reduction_at_least_25_percent": detailed_gates[
            "shadow_full_wall_reduction_at_least_25_percent"
        ],
        "resume_without_gap_or_duplicate": detailed_gates[
            "actual_resume_observed"
        ],
        "output_non_overwrite": detailed_gates[
            "all_successful_outputs_replay_verified"
        ],
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "detailed_gates": detailed_gates,
        "best_batch_size": best,
        "shadow_adapter_id": shadow_adapter_id,
        "comparisons": comparisons,
        "speedups": speedups,
        "shadow_full_wall_time_reduction": wall_reduction,
        "accuracy_withheld": True,
        "gpu_accessed": True,
        "method_effectiveness_claimed": False,
    }
    write_exclusive_or_verify(
        root.resolve() / "qualification_final.json", canonical_json_bytes(report)
    )
    return report
