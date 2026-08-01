"""Create an immutable, CPU-only integrity audit for one formal B=500 run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from safetensors.torch import load_file

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import validate_gsm8k_source_row  # noqa: E402
from eg_sft.evaluation.gsm8k_generation import (  # noqa: E402
    PROMPT_VERSION,
    build_evaluation_prompt,
    score_generation,
)
from eg_sft.experiment.b500_engineering_audit import (  # noqa: E402
    audit_completed_evaluation,
    summarize_adapter_tensors,
)
from eg_sft.experiment.b500_formal_audit import (  # noqa: E402
    audit_checkpoint_directory,
    audit_continuous_temperature_events,
    audit_formal_output_scope,
    audit_thermal_events,
    audit_training_contract,
    compare_tokenizer_texts,
    load_tokenizer_snapshot,
    read_json,
    read_jsonl,
    tokenizer_file_hashes,
    write_json_exclusive,
    write_sha256_sidecar_exclusive,
)
from eg_sft.experiment.b500_matrix import preflight_b500_matrix  # noqa: E402
from eg_sft.experiment.run_manifest import stable_config_hash  # noqa: E402
from eg_sft.training.b500 import (  # noqa: E402
    file_sha256,
    selected_id_sha256,
    tokenize_tulu_candidate,
    validate_selection_manifest,
)


AUDIT_SCHEMA_VERSION = "b500-formal-read-only-audit-v2"
APPROVED_CONTINUOUS_LAUNCHER_SHA256 = (
    "c157cd55c70a564d05266b6f49567854e647aaf4604126a284d318d7c04e7650"
)
AUDIT_IMPLEMENTATION_PATHS = (
    Path("scripts/audit_b500_formal_run.py"),
    Path("src/eg_sft/experiment/b500_formal_audit.py"),
    Path("tests/test_b500_formal_audit.py"),
)


def _git_output(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _audit_code_provenance() -> dict[str, Any]:
    status = _git_output("status", "--porcelain")
    if status:
        raise ValueError("formal audit requires a clean git worktree")
    tracked: dict[str, Any] = {}
    for relative in AUDIT_IMPLEMENTATION_PATHS:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative.as_posix()],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative.as_posix()],
            cwd=ROOT,
            check=True,
        )
        tracked[relative.as_posix()] = {
            "working_file_sha256": file_sha256(ROOT / relative),
            "matches_head": True,
        }
    return {
        "audit_git_commit": _git_output("rev-parse", "HEAD"),
        "git_worktree_clean": True,
        "implementation_files": tracked,
        "command": [sys.executable, *sys.argv],
    }


def _resolve_repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"path leaves repository: {value}") from error
    return path


def _audit_frozen_contract(
    *,
    matrix_path: Path,
    run_manifest: dict[str, Any],
    strategy: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = read_json(matrix_path)
    report = preflight_b500_matrix(
        spec=matrix,
        repo_root=ROOT,
        python_executable=sys.executable,
        matrix_config_path=str(matrix_path.relative_to(ROOT)),
    )
    if report["status"] != "READY_FOR_MANUAL_ONE_JOB_AT_A_TIME":
        raise ValueError("frozen matrix preflight is not ready")
    if stable_config_hash(run_manifest["config"]) != run_manifest["config_hash"]:
        raise ValueError("run manifest effective config hash is internally inconsistent")
    if run_manifest.get("git_is_dirty") is not False:
        raise ValueError("formal run was not launched from a clean worktree")
    subprocess.run(
        ["git", "cat-file", "-e", f"{run_manifest['git_commit']}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    if stable_config_hash(matrix) != run_manifest["config"]["matrix_config_sha256"]:
        raise ValueError("run manifest matrix hash differs from current frozen matrix")
    if report["common_contract_sha256"] != run_manifest["config"]["common_contract_sha256"]:
        raise ValueError("run manifest common contract hash differs from preflight")
    matching = [
        job
        for job in report["jobs"]
        if job["strategy"] == strategy and int(job["seed"]) == seed
    ]
    if len(matching) != 1:
        raise ValueError("run is not one unique frozen matrix job")
    return matrix, report


def _audit_formal_directories(
    *,
    output_root: Path,
    matrix: dict[str, Any],
    current_job: tuple[str, int],
) -> dict[str, Any]:
    actual_jobs: list[tuple[str, int]] = []
    directories: list[str] = []
    for directory in sorted(path for path in output_root.iterdir() if path.is_dir()):
        manifest = read_json(directory / "manifest.json")
        completion = read_json(directory / "run_complete.json")
        if completion.get("status") != "PASS" or completion.get("next_job_started") is not False:
            raise ValueError(f"formal directory is not complete and closed: {directory.name}")
        actual_jobs.append((str(manifest["config"]["strategy"]), int(manifest["seed"])))
        directories.append(directory.name)
    result = audit_formal_output_scope(
        actual_jobs=actual_jobs,
        job_order=matrix["job_order"],
        current_job=current_job,
    )
    result["directories"] = directories
    return result


def _audit_artifact_hashes(
    *,
    run_dir: Path,
    protocol_path: Path,
    recipe_path: Path,
    execution_path: Path,
) -> dict[str, Any]:
    evaluation_dir = run_dir / "evaluation"
    training_dir = run_dir / "training_complete"
    metrics = read_json(evaluation_dir / "metrics.json")
    evaluation_manifest = read_json(evaluation_dir / "manifest.json")
    training_metrics = read_json(training_dir / "training_metrics.json")
    training_artifacts = read_json(training_dir / "artifact_manifest.json")
    completion = read_json(run_dir / "run_complete.json")
    adapter_path = training_dir / "adapter" / "adapter_model.safetensors"
    observed = {
        "protocol_config_sha256": file_sha256(protocol_path),
        "recipe_config_sha256": file_sha256(recipe_path),
        "execution_policy_sha256": file_sha256(execution_path),
        "raw_outputs_sha256": file_sha256(evaluation_dir / "raw_outputs.jsonl"),
        "evaluation_metrics_sha256": file_sha256(evaluation_dir / "metrics.json"),
        "training_metrics_sha256": file_sha256(training_dir / "training_metrics.json"),
        "adapter_model_sha256": file_sha256(adapter_path),
    }
    for key in ("protocol_config_sha256", "recipe_config_sha256", "execution_policy_sha256"):
        if evaluation_manifest.get(key) != observed[key]:
            raise ValueError(f"evaluation manifest hash mismatch: {key}")
    bindings = {
        "raw_outputs_sha256": (
            metrics["raw_outputs_sha256"],
            completion["raw_outputs_sha256"],
        ),
        "evaluation_metrics_sha256": (completion["evaluation_metrics_sha256"],),
        "training_metrics_sha256": (
            completion["training_metrics_sha256"],
            training_artifacts["training_metrics_sha256"],
        ),
        "adapter_model_sha256": (
            metrics["adapter_model_sha256"],
            evaluation_manifest["adapter_model_sha256"],
            training_metrics["adapter_model_sha256"],
            training_artifacts["adapter_model_sha256"],
            completion["adapter_model_sha256"],
        ),
    }
    for key, expected_values in bindings.items():
        if any(value != observed[key] for value in expected_values):
            raise ValueError(f"artifact hash binding mismatch: {key}")
    training_files = {
        "epoch_metrics_sha256": "epoch_metrics.json",
        "training_token_audit_sha256": "training_token_audit.json",
        "development_token_audit_sha256": "development_token_audit.json",
    }
    for key, name in training_files.items():
        if file_sha256(training_dir / name) != training_artifacts[key]:
            raise ValueError(f"training artifact hash mismatch: {name}")
    if completion.get("status") != "PASS" or completion.get("next_job_started") is not False:
        raise ValueError("run_complete does not close exactly one passing job")
    return {**observed, "all_hash_bindings_match": True}


def _audit_runtime_temperature_evidence(
    *,
    run_dir: Path,
    strategy: str,
    seed: int,
    runner_path: Path,
    runner_sha256: str,
    execution: dict[str, Any],
    recorded_max_temperature: float,
    frozen_thermal_report: dict[str, Any],
) -> dict[str, Any]:
    override_path = run_dir / "runtime_policy_overrides.jsonl"
    continuous_path = run_dir / "continuous_temperature_events.jsonl"
    if not override_path.exists() and not continuous_path.exists():
        frozen_max = frozen_thermal_report["max_pause_temperature_c"]
        if frozen_max is not None and recorded_max_temperature != float(frozen_max):
            raise ValueError("run_complete maximum temperature differs from thermal event evidence")
        return {
            "runtime_override_present": False,
            "frozen_policy_followed": True,
            "run_complete_max_temperature_c": recorded_max_temperature,
        }
    if not override_path.is_file() or not continuous_path.is_file():
        raise ValueError("continuous runtime evidence is incomplete")

    overrides = read_jsonl(override_path)
    if len(overrides) != 1:
        raise ValueError("expected exactly one continuous runtime override record")
    override = overrides[0]
    expected_scalars = {
        "event": "continuous_evaluation_runtime_override",
        "strategy": strategy,
        "seed": seed,
        "next_job_launch_allowed": False,
    }
    for key, expected in expected_scalars.items():
        if override.get(key) != expected:
            raise ValueError(f"continuous runtime override mismatch: {key}")
    if override.get("semantic_changes") != []:
        raise ValueError("continuous runtime override reports semantic changes")
    if Path(str(override["resume_run_dir"])).resolve() != run_dir.resolve():
        raise ValueError("continuous runtime override points to a different run")

    recorded_runner = Path(str(override["runner_path"])).resolve()
    if recorded_runner != runner_path.resolve():
        raise ValueError("continuous runtime override runner path mismatch")
    if override.get("runner_sha256") != runner_sha256:
        raise ValueError("continuous runtime override runner hash mismatch")
    if file_sha256(recorded_runner) != runner_sha256:
        raise ValueError("current runner no longer matches continuous override")

    launcher_path = Path(str(override["launcher_path"])).resolve()
    if not launcher_path.is_file():
        raise FileNotFoundError(f"continuous runtime launcher is missing: {launcher_path}")
    launcher_sha256 = str(override.get("launcher_sha256", ""))
    if launcher_sha256 != APPROVED_CONTINUOUS_LAUNCHER_SHA256:
        raise ValueError("continuous runtime launcher hash is not approved")
    if file_sha256(launcher_path) != launcher_sha256:
        raise ValueError("continuous runtime launcher changed after execution")

    changes = override.get("runtime_only_changes")
    if not isinstance(changes, dict):
        raise ValueError("continuous runtime override changes must be an object")
    if float(changes.get("evaluation_inter_example_sleep_seconds", -1)) != 0.0:
        raise ValueError("continuous runtime did not remove the evaluation sleep")
    if changes.get("pause_at_c") != "disabled_during_evaluation":
        raise ValueError("continuous runtime pause disclosure is invalid")
    if changes.get("hardware_firmware_protection") != "unchanged":
        raise ValueError("continuous runtime hardware protection disclosure is invalid")
    sample_every = int(changes["temperature_sample_every_examples"])
    emergency_stop_c = float(changes["software_emergency_stop_c"])
    if emergency_stop_c > 92.0:
        raise ValueError("continuous runtime emergency ceiling exceeds approved value")

    continuous_report = audit_continuous_temperature_events(
        events=read_jsonl(continuous_path),
        sample_every_examples=sample_every,
        emergency_stop_c=emergency_stop_c,
    )
    evidence_maxima = [continuous_report["max_temperature_c"]]
    for key in ("max_pause_temperature_c", "max_resume_temperature_c"):
        value = frozen_thermal_report.get(key)
        if value is not None:
            evidence_maxima.append(float(value))
    combined_max = max(evidence_maxima)
    if recorded_max_temperature != combined_max:
        raise ValueError("run_complete maximum temperature differs from combined evidence")
    return {
        "runtime_override_present": True,
        "frozen_policy_followed": False,
        "semantic_fields_changed": False,
        "runtime_efficiency_comparison_allowed": False,
        "launcher_path": str(launcher_path),
        "launcher_sha256": launcher_sha256,
        "override_log_sha256": file_sha256(override_path),
        "continuous_temperature_log_sha256": file_sha256(continuous_path),
        "original_hard_stop_at_c": float(execution["temperature"]["hard_stop_at_c"]),
        "run_complete_max_temperature_c": recorded_max_temperature,
        "continuous_temperature": continuous_report,
        "claim_boundary": (
            "Accuracy artifacts remain auditable, but runtime speed, thermal, and "
            "efficiency values are not comparable with frozen-policy jobs."
        ),
    }


def _audit_evaluation(
    *,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    data_manifest_dir: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    evaluation_dir = run_dir / "evaluation"
    metrics = read_json(evaluation_dir / "metrics.json")
    rows = read_jsonl(evaluation_dir / "raw_outputs.jsonl")
    records = [
        row
        for row in read_jsonl(data_manifest_dir / "gsm8k_records.jsonl")
        if row["protocol_split"] == recipe["evaluation"]["split"]
    ]
    records.sort(key=lambda row: (int(row["source_index"]), str(row["record_id"])))
    row_report = audit_completed_evaluation(
        rows=rows,
        frozen_records=records,
        metrics=metrics,
        prompt_version=PROMPT_VERSION,
    )
    gsm_config = protocol["datasets"]["gsm8k"]
    gsm_test = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        split="test",
        revision=gsm_config["revision"],
    )
    if len(gsm_test) != len(records):
        raise ValueError("pinned GSM8K test count differs from frozen records")
    for index, (source, record, saved) in enumerate(zip(gsm_test, records, rows, strict=True)):
        validate_gsm8k_source_row(record, source)
        recomputed = score_generation(
            record=record,
            gold_answer_text=source["answer"],
            generated_text=saved["raw_output"],
        )
        if recomputed != saved:
            raise ValueError(f"raw output does not independently rescore at row {index}")
    return (
        {
            **row_report,
            "pinned_dataset_rescore_count": len(rows),
            "all_raw_outputs_rescore_exactly": True,
        },
        rows,
        gsm_test,
    )


def _audit_tokenizer_equivalence(
    *,
    reference_tokenizer_dir: Path,
    saved_tokenizer_dir: Path,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    selected: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    gsm_test: Any,
) -> dict[str, Any]:
    revision = str(protocol["model"]["revision"])
    if reference_tokenizer_dir.name != revision:
        raise ValueError("reference tokenizer snapshot directory does not match model revision")
    reference = load_tokenizer_snapshot(reference_tokenizer_dir)
    saved = load_tokenizer_snapshot(saved_tokenizer_dir)
    if reference.special_tokens_map != saved.special_tokens_map:
        raise ValueError("saved tokenizer special-token map differs from the reference")
    attributes = (
        "vocab_size",
        "eos_token_id",
        "pad_token_id",
        "bos_token_id",
        "unk_token_id",
        "truncation_side",
    )
    for name in attributes:
        if getattr(reference, name) != getattr(saved, name):
            raise ValueError(f"saved tokenizer attribute differs: {name}")
    evaluation_prompts = [build_evaluation_prompt(row["question"]) for row in gsm_test]
    prompt_report = compare_tokenizer_texts(
        reference=reference,
        saved=saved,
        texts=evaluation_prompts,
        tokenizer_kwargs={
            "padding": True,
            "truncation": True,
            "max_length": int(recipe["evaluation"]["max_input_length"]),
        },
        single_item_batch=True,
    )
    raw_report = compare_tokenizer_texts(
        reference=reference,
        saved=saved,
        texts=[str(row["raw_output"]) for row in evaluation_rows],
        tokenizer_kwargs={"add_special_tokens": False},
        single_item_batch=False,
    )
    candidate_config = protocol["datasets"]["candidate_pool"]
    tulu = load_dataset(
        candidate_config["repo_id"],
        candidate_config["config"],
        split="train",
        revision=candidate_config["revision"],
    )
    training_mismatches: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        source = tulu[int(candidate["source_index"])]
        reference_example, reference_audit = tokenize_tulu_candidate(
            tokenizer=reference,
            candidate=candidate,
            raw_row=source,
            max_length=int(recipe["training"]["max_length"]),
        )
        saved_example, saved_audit = tokenize_tulu_candidate(
            tokenizer=saved,
            candidate=candidate,
            raw_row=source,
            max_length=int(recipe["training"]["max_length"]),
        )
        expected_counts = {
            "total_tokens": int(candidate["total_tokens"]),
            "supervised_tokens": int(candidate["supervised_tokens"]),
        }
        count_mismatch = any(
            int(reference_audit[key]) != value or int(saved_audit[key]) != value
            for key, value in expected_counts.items()
        )
        if reference_example != saved_example or reference_audit != saved_audit or count_mismatch:
            training_mismatches.append(
                {"index": index, "candidate_id": candidate["candidate_id"]}
            )
    if training_mismatches:
        raise ValueError(
            f"saved tokenizer changes {len(training_mismatches)} training examples; "
            f"first={training_mismatches[0]}"
        )
    return {
        "status": "PASS",
        "loader": "local_tokenizer_json_plus_pinned_tokenizer_config",
        "reference_snapshot_revision": revision,
        "reference_tokenizer_files": tokenizer_file_hashes(reference_tokenizer_dir),
        "saved_tokenizer_files": tokenizer_file_hashes(saved_tokenizer_dir),
        "special_tokens_map_equal": True,
        "attributes_equal": {name: getattr(reference, name) for name in attributes},
        "evaluation_prompts": prompt_report,
        "saved_raw_outputs": raw_report,
        "training_candidates": {
            "compared_count": len(selected),
            "tokenized_example_mismatch_count": 0,
            "exact_input_attention_and_label_equality": True,
        },
    }


def _audit_invocations(run_dir: Path) -> dict[str, Any]:
    invocations = read_jsonl(run_dir / "invocations.jsonl")
    failures = read_jsonl(run_dir / "failures.jsonl") if (run_dir / "failures.jsonl").exists() else []
    event_counts = Counter(str(row.get("event")) for row in invocations)
    if event_counts["invocation_complete"] != 1:
        raise ValueError("formal run must contain exactly one invocation_complete event")
    if invocations[-1].get("event") != "invocation_complete":
        raise ValueError("the final invocation event is not invocation_complete")
    return {
        "invocation_event_counts": dict(sorted(event_counts.items())),
        "failure_count": len(failures),
        "failure_types": dict(
            sorted(Counter(str(row.get("error_type")) for row in failures).items())
        ),
        "final_invocation_complete": True,
        "non_blocking_note": (
            "Completed-run hashes, checkpoint bindings and the ordered evaluation prefix are "
            "authoritative; interrupted invocations remain visible in this count."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=Path("configs/b500_formal_matrix_v1.json"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-checkpoint-payload-hashes",
        action="store_true",
        help="Development-only shortcut; artifacts record that payload hashes were not verified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix_config.resolve()
    run_dir = args.run_dir.resolve()
    reference_tokenizer_dir = args.reference_tokenizer_dir.resolve()
    output_path = args.output.resolve()
    if output_path.exists() or output_path.with_name(output_path.name + ".sha256").exists():
        raise FileExistsError(f"audit output or sidecar already exists: {output_path}")
    if output_path.suffix.lower() != ".json":
        raise ValueError("audit output must be a JSON file")
    if output_path.parent != run_dir / "audits":
        raise ValueError("audit output must be a direct child of the run's audits directory")

    code_provenance = _audit_code_provenance()
    run_manifest = read_json(run_dir / "manifest.json")
    strategy = str(run_manifest["config"]["strategy"])
    seed = int(run_manifest["seed"])
    matrix, preflight = _audit_frozen_contract(
        matrix_path=matrix_path,
        run_manifest=run_manifest,
        strategy=strategy,
        seed=seed,
    )
    protocol_path = _resolve_repo_path(matrix["protocol_config"]["path"])
    recipe_path = _resolve_repo_path(matrix["recipe_config"]["path"])
    execution_path = _resolve_repo_path(matrix["execution_config"]["path"])
    data_manifest_dir = _resolve_repo_path(matrix["data_manifest"]["directory"])
    selection_path = _resolve_repo_path(matrix["selections"][strategy]["path"])
    protocol = read_json(protocol_path)
    recipe = read_json(recipe_path)
    execution = read_json(execution_path)
    selection_manifest = read_json(selection_path)
    selected = validate_selection_manifest(
        selection_manifest,
        expected_strategy=strategy,
        expected_budget=int(recipe["selection"]["budget"]),
        expected_selection_seed=int(recipe["selection"]["selection_seed"]),
    )
    if selected_id_sha256(selected) != run_manifest["config"]["selected_id_sha256"]:
        raise ValueError("selected candidate ID hash differs from run manifest")
    if file_sha256(selection_path) != run_manifest["config"]["selection_manifest_sha256"]:
        raise ValueError("selection manifest file hash differs from run manifest")

    artifact_hashes = _audit_artifact_hashes(
        run_dir=run_dir,
        protocol_path=protocol_path,
        recipe_path=recipe_path,
        execution_path=execution_path,
    )
    evaluation_report, evaluation_rows, gsm_test = _audit_evaluation(
        protocol=protocol,
        recipe=recipe,
        data_manifest_dir=data_manifest_dir,
        run_dir=run_dir,
    )
    adapter_path = run_dir / "training_complete" / "adapter" / "adapter_model.safetensors"
    adapter_report = summarize_adapter_tensors(load_file(adapter_path))
    training_metrics = read_json(run_dir / "training_complete" / "training_metrics.json")
    with (run_dir / "training_complete" / "training_token_audit.json").open(
        encoding="utf-8"
    ) as handle:
        token_audit = json.load(handle)
    if not isinstance(token_audit, list):
        raise ValueError("training_token_audit.json must be a list")
    training_report = audit_training_contract(
        metrics=training_metrics,
        token_audit=token_audit,
        selected=selected,
        training_config=recipe["training"],
        strategy=strategy,
        seed=seed,
        adapter_parameter_count=int(adapter_report["total_parameters"]),
    )
    total_micro_batches = len(selected) * int(recipe["training"]["epochs"])
    optimizer_steps = int(training_metrics["optimizer_steps_planned"])
    checkpoint_report = audit_checkpoint_directory(
        checkpoint_directory=run_dir / "checkpoints",
        expected_binding={
            "git_commit": run_manifest["git_commit"],
            "run_config_hash": run_manifest["config_hash"],
            "strategy": strategy,
            "seed": seed,
            "selected_id_sha256": run_manifest["config"]["selected_id_sha256"],
        },
        total_micro_batches=total_micro_batches,
        gradient_accumulation_steps=int(recipe["training"]["gradient_accumulation_steps"]),
        optimizer_steps=optimizer_steps,
        verify_payload_hashes=not args.skip_checkpoint_payload_hashes,
    )
    if args.skip_checkpoint_payload_hashes:
        raise ValueError("formal PASS artifact requires full checkpoint payload hashing")
    thermal_report = audit_thermal_events(
        events=read_jsonl(run_dir / "thermal_events.jsonl"),
        pause_at_c=float(execution["temperature"]["pause_at_c"]),
        resume_at_c=float(execution["temperature"]["resume_at_c"]),
        hard_stop_at_c=float(execution["temperature"]["hard_stop_at_c"]),
    )
    recorded_max_temperature = float(
        read_json(run_dir / "run_complete.json")["max_temperature_c_observed_by_runner"]
    )
    runtime_temperature_report = _audit_runtime_temperature_evidence(
        run_dir=run_dir,
        strategy=strategy,
        seed=seed,
        runner_path=_resolve_repo_path(matrix["runner"]["path"]),
        runner_sha256=str(matrix["runner"]["sha256"]),
        execution=execution,
        recorded_max_temperature=recorded_max_temperature,
        frozen_thermal_report=thermal_report,
    )
    thermal_report["run_complete_max_temperature_c"] = recorded_max_temperature
    tokenizer_report = _audit_tokenizer_equivalence(
        reference_tokenizer_dir=reference_tokenizer_dir,
        saved_tokenizer_dir=run_dir / "training_complete" / "tokenizer",
        protocol=protocol,
        recipe=recipe,
        selected=selected,
        evaluation_rows=evaluation_rows,
        gsm_test=gsm_test,
    )
    output_scope = _audit_formal_directories(
        output_root=_resolve_repo_path(matrix["output_root"]),
        matrix=matrix,
        current_job=(strategy, seed),
    )
    invocations = _audit_invocations(run_dir)

    payload = {
        "status": "PASS",
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_type": "deterministic_formal_run_closure",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source_run_id": run_manifest["run_id"],
        "source_run_git_commit": run_manifest["git_commit"],
        "strategy": strategy,
        "seed": seed,
        "code_provenance": code_provenance,
        "frozen_matrix": {
            "matrix_version": preflight["matrix_version"],
            "matrix_config_sha256": preflight["matrix_config_sha256"],
            "common_contract_sha256": preflight["common_contract_sha256"],
            "all_three_selections_ready": preflight["ready_selection_count"] == 3,
        },
        "selection": {
            "manifest_path": str(selection_path),
            "manifest_sha256": file_sha256(selection_path),
            "selected_id_sha256": selected_id_sha256(selected),
            "selected_count": len(selected),
        },
        "training": training_report,
        "serialized_adapter": adapter_report,
        "evaluation": evaluation_report,
        "tokenizer_equivalence": tokenizer_report,
        "checkpoints": checkpoint_report,
        "thermal": thermal_report,
        "runtime_temperature_evidence": runtime_temperature_report,
        "artifact_hashes": artifact_hashes,
        "invocations": invocations,
        "formal_output_scope": output_scope,
        "evaluation_type": "real_gt",
        "claim_boundary": (
            "This artifact verifies one completed formal B=500 job. It does not compare "
            "selectors, estimate seed variance, or support a matrix-level conclusion."
        ),
        "write_policy": {
            "experiment_inputs_mutated": False,
            "only_new_files": [str(output_path), str(output_path) + ".sha256"],
            "overwrite_permitted": False,
        },
    }
    write_json_exclusive(output_path, payload)
    sidecar = write_sha256_sidecar_exclusive(output_path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact": str(output_path),
                "artifact_sha256": file_sha256(output_path),
                "sha256_sidecar": str(sidecar),
                "strategy": strategy,
                "seed": seed,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
