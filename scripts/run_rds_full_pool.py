"""Run resumable, thermal-gated full-pool RDS+ representation scoring.

The command has four explicit stages:

1. ``prepare`` audits the frozen source rows and response-trainability on CPU.
2. ``encode`` computes exactly one immutable query or candidate embedding chunk.
3. ``status`` validates completed chunks without loading a model.
4. ``finalize`` joins complete chunks and computes frozen all/error RDS+ ranks.

No stage trains a model or evaluates downstream GSM8K accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.data.public_gsm8k import (  # noqa: E402
    sha256_text,
    validate_gsm8k_source_row,
)
from eg_sft.experiment.rds_full_pool import (  # noqa: E402
    build_score_rows,
    canonical_json_sha256,
    chunk_bounds,
    chunk_count,
    chunk_manifest_filename,
    ordered_value_sha256,
    tensor_sha256,
    validate_chunk_manifest,
)
from eg_sft.selection.h1a_sample import (  # noqa: E402
    stratified_candidate_sample,
)
from eg_sft.selection.query_groups import load_jsonl  # noqa: E402
from eg_sft.selection.rds import (  # noqa: E402
    RDS_FORMAT_VERSION,
    encode_rds_texts,
    format_gsm8k_rds_text,
    format_tulu_rds_text,
)
from eg_sft.training.b500 import (  # noqa: E402
    file_sha256,
    tokenize_tulu_candidate,
    validate_candidate_source,
)


RUN_CONTRACT_SCHEMA = "rds-full-pool-run-contract-v1"
PREPARED_SCHEMA = "rds-full-pool-prepared-v1"
CHUNK_SCHEMA = "rds-full-pool-embedding-chunk-v1"
FINALIZATION_SCHEMA = "rds-full-pool-finalization-v1"

IMPLEMENTATION_PATHS = (
    "scripts/run_rds_full_pool.py",
    "src/eg_sft/experiment/rds_full_pool.py",
    "src/eg_sft/selection/rds.py",
    "src/eg_sft/selection/h1a_sample.py",
    "src/eg_sft/training/b500.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _relative_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside repository: {resolved}") from error


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _git_status() -> str:
    process = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout


def _file_binding(path: Path) -> dict[str, str]:
    return {
        "path": _relative_repo_path(path),
        "sha256": file_sha256(path),
    }


def _resolve_binding(binding: dict[str, Any]) -> Path:
    relative = Path(str(binding["path"]))
    if relative.is_absolute():
        raise ValueError("run-contract paths must be repository-relative")
    resolved = (ROOT / relative).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("run-contract path escapes the repository") from error
    if not resolved.is_file():
        raise ValueError(f"bound file is missing: {relative.as_posix()}")
    observed = file_sha256(resolved)
    if observed != binding.get("sha256"):
        raise ValueError(
            f"bound file changed: {relative.as_posix()} "
            f"(observed {observed}, expected {binding.get('sha256')})"
        )
    return resolved


def _tokenizer(
    *,
    model_config: dict[str, Any],
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        use_fast=True,
    )
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _nvidia_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        (
            "--query-gpu=name,temperature.gpu,memory.used,memory.total,"
            "utilization.gpu,power.draw"
        ),
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
    rows = list(csv.reader(process.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeError("expected exactly one visible NVIDIA GPU")
    values = [value.strip() for value in rows[0]]
    return {
        "name": values[0],
        "temperature_c": int(float(values[1])),
        "memory_used_mib": int(float(values[2])),
        "memory_total_mib": int(float(values[3])),
        "utilization_percent": int(float(values[4])),
        "power_draw_w": float(values[5]),
        "observed_at_utc": datetime.now(UTC).isoformat(),
    }


def _system_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "available_memory_bytes": int(memory.available),
        "available_memory_gib": memory.available / 1024**3,
        "total_memory_bytes": int(memory.total),
        "cpu_percent": float(psutil.cpu_percent(interval=0.2)),
    }


def _load_contract(run_dir: Path) -> dict[str, Any]:
    contract_path = run_dir / "run_contract.json"
    contract = _read_json(contract_path)
    if contract.get("schema_version") != RUN_CONTRACT_SCHEMA:
        raise ValueError("run contract schema changed")
    claimed = contract.get("run_contract_sha256")
    hash_payload = dict(contract)
    hash_payload.pop("run_contract_sha256", None)
    observed = canonical_json_sha256(hash_payload)
    if claimed != observed:
        raise ValueError("run contract self-hash changed")
    for binding in contract["input_bindings"].values():
        _resolve_binding(binding)
    for binding in contract["implementation_bindings"].values():
        _resolve_binding(binding)
    return contract


def _load_prepared(
    run_dir: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared_path = run_dir / "prepared.json"
    prepared = _read_json(prepared_path)
    if prepared.get("schema_version") != PREPARED_SCHEMA:
        raise ValueError("prepared artifact schema changed")
    if prepared.get("run_contract_sha256") != contract["run_contract_sha256"]:
        raise ValueError("prepared artifact belongs to another run contract")
    candidate_path = run_dir / str(prepared["candidate_inventory"]["path"])
    query_path = run_dir / str(prepared["query_inventory"]["path"])
    if file_sha256(candidate_path) != prepared["candidate_inventory"]["sha256"]:
        raise ValueError("candidate inventory hash changed")
    if file_sha256(query_path) != prepared["query_inventory"]["sha256"]:
        raise ValueError("query inventory hash changed")
    candidates = load_jsonl(candidate_path)
    queries = load_jsonl(query_path)
    if len(candidates) != int(prepared["audited_candidate_count"]):
        raise ValueError("candidate inventory row count changed")
    if len(queries) != int(prepared["all_query_count"]):
        raise ValueError("query inventory row count changed")
    eligible = [
        row for row in candidates if bool(row["response_only_trainable"])
    ]
    if len(eligible) != int(prepared["eligible_candidate_count"]):
        raise ValueError("eligible candidate count changed")
    if [int(row["eligible_index"]) for row in eligible] != list(
        range(len(eligible))
    ):
        raise ValueError("eligible candidate order changed")
    return prepared, candidates, queries


def _prepare(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    protocol_path = args.protocol_config.resolve()
    execution_path = args.execution_config.resolve()
    data_dir = args.data_manifest_dir.resolve()
    query_dir = args.query_groups_dir.resolve()
    protocol = _read_json(protocol_path)
    execution = _read_json(execution_path)
    representation = execution["representation"]
    selection = execution["selection"]
    if representation["version"] != RDS_FORMAT_VERSION:
        raise ValueError("RDS representation version changed")
    if tuple(selection["strategies"]) != ("rds_all", "rds_error"):
        raise ValueError("selector definitions changed")
    if int(selection["selection_seed"]) != int(protocol["seed"]):
        raise ValueError("selection seed changed")
    if int(representation["max_length"]) != 512:
        raise ValueError("the frozen response-trainability length changed")
    if bool(representation["tokenizer_fix_mistral_regex"]):
        raise ValueError("tokenizer compatibility mode differs from frozen H1a")

    candidate_pool_path = data_dir / "tulu_candidate_pool.jsonl"
    gsm_records_path = data_dir / "gsm8k_records.jsonl"
    all_queries_path = query_dir / "all_queries.jsonl"
    error_queries_path = query_dir / "error_queries.jsonl"
    query_manifest_path = query_dir / "query_group_manifest.json"
    required_paths = (
        protocol_path,
        execution_path,
        candidate_pool_path,
        gsm_records_path,
        all_queries_path,
        error_queries_path,
        query_manifest_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    pool = load_jsonl(candidate_pool_path)
    expected_pool_size = int(protocol["candidate_pool_size"])
    if len(pool) != expected_pool_size:
        raise ValueError("frozen Tulu candidate pool is not exactly 10,000 rows")
    ordered_pool = stratified_candidate_sample(
        pool,
        count=len(pool),
        seed=int(selection["selection_seed"]),
    )
    all_queries_full = load_jsonl(all_queries_path)
    error_queries_full = load_jsonl(error_queries_path)
    query_manifest = _read_json(query_manifest_path)
    if len(all_queries_full) != int(query_manifest["all_query_count"]):
        raise ValueError("all-query count changed")
    if len(error_queries_full) != int(query_manifest["error_query_count"]):
        raise ValueError("error-query count changed")
    error_ids_full = {str(row["record_id"]) for row in error_queries_full}
    expected_error_ids = {
        str(row["record_id"])
        for row in all_queries_full
        if not bool(row["numeric_correct"])
    }
    if error_ids_full != expected_error_ids:
        raise ValueError("error-query file no longer matches frozen labels")

    if args.smoke:
        if args.candidate_limit is None or args.query_limit is None:
            raise ValueError("smoke preparation requires both limits")
        if not 1 <= args.candidate_limit < expected_pool_size:
            raise ValueError("invalid smoke candidate limit")
        if not 1 <= args.query_limit < len(all_queries_full):
            raise ValueError("invalid smoke query limit")
        ordered_pool = ordered_pool[: args.candidate_limit]
        all_queries = all_queries_full[: args.query_limit]
        scope = "engineering_smoke"
    else:
        if args.candidate_limit is not None or args.query_limit is not None:
            raise ValueError("formal preparation does not accept scope limits")
        all_queries = all_queries_full
        scope = "formal_10000_candidate_pool"
    error_ids = {
        str(row["record_id"])
        for row in all_queries
        if not bool(row["numeric_correct"])
    }
    if not error_ids:
        raise ValueError("selected query scope has no error-conditioned queries")

    model_config = protocol["model"]
    candidate_config = protocol["datasets"]["candidate_pool"]
    gsm_config = protocol["datasets"]["gsm8k"]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(execution["thermal"]["cpu_threads"]))
    tokenizer = _tokenizer(model_config=model_config)
    tulu = load_dataset(
        candidate_config["repo_id"],
        candidate_config["config"],
        split="train",
        revision=candidate_config["revision"],
    )
    gsm = load_dataset(
        gsm_config["repo_id"],
        gsm_config["config"],
        split="train",
        revision=gsm_config["revision"],
    )
    gsm_records = {
        str(row["record_id"]): row for row in load_jsonl(gsm_records_path)
    }

    candidate_inventory: list[dict[str, Any]] = []
    eligible_index = 0
    started = time.perf_counter()
    for order_index, candidate in enumerate(ordered_pool):
        raw_row = tulu[int(candidate["source_index"])]
        messages = validate_candidate_source(candidate, raw_row)
        rds_text = format_tulu_rds_text(
            messages,
            eos_token=tokenizer.eos_token,
        )
        try:
            _, token_audit = tokenize_tulu_candidate(
                tokenizer=tokenizer,
                candidate=candidate,
                raw_row=raw_row,
                max_length=int(representation["max_length"]),
            )
            trainable = int(token_audit["supervised_tokens"]) > 0
        except ValueError as error:
            if "response was fully truncated" not in str(error):
                raise
            token_audit = {
                "total_tokens": int(representation["max_length"]),
                "supervised_tokens": 0,
            }
            trainable = False
        row = {
            **candidate,
            "candidate_order_index": order_index,
            "eligible_index": eligible_index if trainable else None,
            "response_only_trainable": trainable,
            "total_tokens": int(token_audit["total_tokens"]),
            "supervised_tokens": int(token_audit["supervised_tokens"]),
            "rds_text_sha256": sha256_text(rds_text),
        }
        candidate_inventory.append(row)
        if trainable:
            eligible_index += 1
        if (order_index + 1) % 1000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "prepare_candidates",
                        "audited": order_index + 1,
                        "eligible": eligible_index,
                    }
                ),
                flush=True,
            )

    query_inventory: list[dict[str, Any]] = []
    for query_index, query in enumerate(all_queries):
        record_id = str(query["record_id"])
        if record_id not in gsm_records:
            raise ValueError(f"missing frozen GSM8K record: {record_id}")
        frozen_record = gsm_records[record_id]
        if int(frozen_record["source_index"]) != int(query["source_index"]):
            raise ValueError(f"query source index changed: {record_id}")
        raw_row = gsm[int(query["source_index"])]
        validate_gsm8k_source_row(frozen_record, raw_row)
        rds_text = format_gsm8k_rds_text(
            question=str(raw_row["question"]),
            answer=str(raw_row["answer"]),
            eos_token=tokenizer.eos_token,
        )
        query_inventory.append(
            {
                "query_index": query_index,
                "record_id": record_id,
                "source_index": int(query["source_index"]),
                "question_sha256": query["question_sha256"],
                "is_error_query": record_id in error_ids,
                "rds_text_sha256": sha256_text(rds_text),
            }
        )

    input_bindings = {
        "protocol_config": _file_binding(protocol_path),
        "execution_config": _file_binding(execution_path),
        "candidate_pool": _file_binding(candidate_pool_path),
        "gsm8k_records": _file_binding(gsm_records_path),
        "all_queries": _file_binding(all_queries_path),
        "error_queries": _file_binding(error_queries_path),
        "query_group_manifest": _file_binding(query_manifest_path),
    }
    implementation_bindings = {
        path: _file_binding(ROOT / path) for path in IMPLEMENTATION_PATHS
    }
    contract: dict[str, Any] = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "scope": scope,
        "representation": representation,
        "thermal": execution["thermal"],
        "selection": selection,
        "protocol": {
            "seed": int(protocol["seed"]),
            "model": model_config,
            "datasets": protocol["datasets"],
            "candidate_pool_size": expected_pool_size,
        },
        "prepared_candidate_scope_count": len(ordered_pool),
        "prepared_query_scope_count": len(all_queries),
        "source_git_commit": _git_commit(),
        "git_status_at_prepare": _git_status(),
        "input_bindings": input_bindings,
        "implementation_bindings": implementation_bindings,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "claim_boundary": execution["claim_boundary"],
    }
    contract["run_contract_sha256"] = canonical_json_sha256(contract)

    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "run_contract.json", contract)
    _write_jsonl(run_dir / "candidate_inventory.jsonl", candidate_inventory)
    _write_jsonl(run_dir / "query_inventory.jsonl", query_inventory)
    eligible_candidates = [
        row for row in candidate_inventory if row["response_only_trainable"]
    ]
    prepared = {
        "schema_version": PREPARED_SCHEMA,
        "status": "COMPLETE",
        "run_contract_sha256": contract["run_contract_sha256"],
        "scope": scope,
        "audited_candidate_count": len(candidate_inventory),
        "eligible_candidate_count": len(eligible_candidates),
        "excluded_fully_truncated_count": (
            len(candidate_inventory) - len(eligible_candidates)
        ),
        "all_query_count": len(query_inventory),
        "error_query_count": sum(
            bool(row["is_error_query"]) for row in query_inventory
        ),
        "candidate_inventory": {
            "path": "candidate_inventory.jsonl",
            "sha256": file_sha256(run_dir / "candidate_inventory.jsonl"),
            "ordered_candidate_id_sha256": ordered_value_sha256(
                [str(row["candidate_id"]) for row in candidate_inventory]
            ),
            "ordered_eligible_id_sha256": ordered_value_sha256(
                [str(row["candidate_id"]) for row in eligible_candidates]
            ),
        },
        "query_inventory": {
            "path": "query_inventory.jsonl",
            "sha256": file_sha256(run_dir / "query_inventory.jsonl"),
            "ordered_query_id_sha256": ordered_value_sha256(
                [str(row["record_id"]) for row in query_inventory]
            ),
        },
        "query_chunk_count": chunk_count(
            len(query_inventory),
            int(representation["query_chunk_size"]),
        ),
        "candidate_chunk_count": chunk_count(
            len(eligible_candidates),
            int(representation["candidate_chunk_size"]),
        ),
        "preparation_elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(run_dir / "prepared.json", prepared)
    print(json.dumps({"run_dir": str(run_dir), **prepared}, indent=2))


def _inventory_for_kind(
    *,
    kind: str,
    candidates: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if kind == "query":
        return list(queries), "record_id"
    if kind == "candidate":
        return [
            row for row in candidates if row["response_only_trainable"]
        ], "candidate_id"
    raise ValueError("kind must be query or candidate")


def _chunk_directory(run_dir: Path, kind: str) -> Path:
    return run_dir / "embedding_chunks" / kind


def _load_chunk(
    *,
    run_dir: Path,
    contract: dict[str, Any],
    kind: str,
    chunk_index: int,
    expected_rows: Sequence[dict[str, Any]],
    id_field: str,
    deep: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    directory = _chunk_directory(run_dir, kind)
    manifest_path = directory / chunk_manifest_filename(chunk_index)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    artifact_path = directory / str(manifest.get("artifact_file", ""))
    if not artifact_path.is_file():
        raise ValueError(f"chunk artifact is missing: {artifact_path}")
    artifact_sha256 = file_sha256(artifact_path)
    expected_ids = [str(row[id_field]) for row in expected_rows]
    validate_chunk_manifest(
        manifest=manifest,
        expected_kind=kind,
        expected_chunk_index=chunk_index,
        expected_ids=expected_ids,
        expected_representation_version=RDS_FORMAT_VERSION,
        expected_run_contract_sha256=contract["run_contract_sha256"],
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
    )
    payload: dict[str, Any] | None = None
    if deep:
        loaded = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(loaded, dict):
            raise ValueError("chunk artifact payload is not a dictionary")
        payload = loaded
        if loaded.get("ids") != expected_ids:
            raise ValueError("chunk artifact IDs changed")
        embeddings = loaded.get("embeddings")
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("chunk artifact has no embedding tensor")
        if list(embeddings.shape) != manifest.get("embedding_shape"):
            raise ValueError("chunk embedding shape changed")
        if str(embeddings.dtype) != manifest.get("embedding_dtype"):
            raise ValueError("chunk embedding dtype changed")
        if tensor_sha256(embeddings) != manifest.get("embedding_tensor_sha256"):
            raise ValueError("chunk embedding tensor hash changed")
    return manifest, payload


def _reconstruct_texts(
    *,
    kind: str,
    rows: Sequence[dict[str, Any]],
    contract: dict[str, Any],
    tokenizer: Any,
) -> list[str]:
    dataset_config = contract["protocol"]["datasets"]
    if kind == "candidate":
        config = dataset_config["candidate_pool"]
        dataset = load_dataset(
            config["repo_id"],
            config["config"],
            split="train",
            revision=config["revision"],
        )
        texts: list[str] = []
        for row in rows:
            raw_row = dataset[int(row["source_index"])]
            messages = validate_candidate_source(row, raw_row)
            text = format_tulu_rds_text(
                messages,
                eos_token=tokenizer.eos_token,
            )
            if sha256_text(text) != row["rds_text_sha256"]:
                raise ValueError(f"RDS text changed for {row['candidate_id']}")
            texts.append(text)
        return texts

    config = dataset_config["gsm8k"]
    dataset = load_dataset(
        config["repo_id"],
        config["config"],
        split="train",
        revision=config["revision"],
    )
    texts = []
    for row in rows:
        raw_row = dataset[int(row["source_index"])]
        if sha256_text(str(raw_row["question"])) != row["question_sha256"]:
            raise ValueError(f"GSM8K question changed for {row['record_id']}")
        text = format_gsm8k_rds_text(
            question=str(raw_row["question"]),
            answer=str(raw_row["answer"]),
            eos_token=tokenizer.eos_token,
        )
        if sha256_text(text) != row["rds_text_sha256"]:
            raise ValueError(f"RDS text changed for {row['record_id']}")
        texts.append(text)
    return texts


def _encode(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("RDS encoding requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("RDS encoding requires BF16 support")
    run_dir = args.run_dir.resolve()
    contract = _load_contract(run_dir)
    prepared, candidates, queries = _load_prepared(run_dir, contract)
    inventory, id_field = _inventory_for_kind(
        kind=args.kind,
        candidates=candidates,
        queries=queries,
    )
    chunk_size = int(
        contract["representation"][
            "query_chunk_size"
            if args.kind == "query"
            else "candidate_chunk_size"
        ]
    )
    start, end = chunk_bounds(len(inventory), chunk_size, args.chunk_index)
    rows = inventory[start:end]
    directory = _chunk_directory(run_dir, args.kind)
    directory.mkdir(parents=True, exist_ok=True)
    fixed_manifest = directory / chunk_manifest_filename(args.chunk_index)
    if fixed_manifest.is_file():
        manifest, _ = _load_chunk(
            run_dir=run_dir,
            contract=contract,
            kind=args.kind,
            chunk_index=args.chunk_index,
            expected_rows=rows,
            id_field=id_field,
            deep=True,
        )
        print(
            json.dumps(
                {
                    "status": "ALREADY_COMPLETE",
                    "manifest": str(fixed_manifest),
                    "embedding_tensor_sha256": manifest[
                        "embedding_tensor_sha256"
                    ],
                },
                indent=2,
            )
        )
        return

    thermal = contract["thermal"]
    gpu_start = _nvidia_snapshot()
    system_start = _system_snapshot()
    if gpu_start["temperature_c"] > int(thermal["max_start_temperature_c"]):
        raise RuntimeError(
            "GPU is too warm to start: "
            f"{gpu_start['temperature_c']} C > "
            f"{thermal['max_start_temperature_c']} C"
        )
    max_start_memory = int(thermal.get("maximum_start_memory_mib", 512))
    if gpu_start["memory_used_mib"] > max_start_memory:
        raise RuntimeError(
            "GPU memory is not idle enough to start: "
            f"{gpu_start['memory_used_mib']} MiB > {max_start_memory} MiB"
        )
    if system_start["available_memory_gib"] < float(
        thermal["minimum_free_system_memory_gib"]
    ):
        raise RuntimeError("not enough free system memory to start")

    model_config = contract["protocol"]["model"]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(thermal["cpu_threads"]))
    set_seed(int(contract["selection"]["selection_seed"]))
    tokenizer = _tokenizer(model_config=model_config)
    texts = _reconstruct_texts(
        kind=args.kind,
        rows=rows,
        contract=contract,
        tokenizer=tokenizer,
    )
    device = torch.device("cuda")
    samples: list[dict[str, Any]] = [gpu_start]
    sample_every = int(thermal["temperature_sample_every_batches"])
    abort_temperature = int(thermal["abort_temperature_c"])

    def thermal_callback(batch_index: int, batch_total: int) -> None:
        if batch_index % sample_every != 0 and batch_index != batch_total:
            return
        snapshot = _nvidia_snapshot()
        snapshot["batch_index"] = batch_index
        snapshot["batch_total"] = batch_total
        samples.append(snapshot)
        print(
            json.dumps(
                {
                    "stage": "thermal_sample",
                    "kind": args.kind,
                    "chunk_index": args.chunk_index,
                    "batch": f"{batch_index}/{batch_total}",
                    "temperature_c": snapshot["temperature_c"],
                    "memory_used_mib": snapshot["memory_used_mib"],
                    "power_draw_w": snapshot["power_draw_w"],
                }
            ),
            flush=True,
        )
        if snapshot["temperature_c"] >= abort_temperature:
            raise RuntimeError(
                f"thermal abort at {snapshot['temperature_c']} C"
            )

    model: Any = None
    embeddings: torch.Tensor | None = None
    started = time.perf_counter()
    try:
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        after_load = _nvidia_snapshot()
        after_load["event"] = "model_loaded"
        samples.append(after_load)
        if after_load["temperature_c"] >= abort_temperature:
            raise RuntimeError(
                f"thermal abort after model load at "
                f"{after_load['temperature_c']} C"
            )
        torch.cuda.reset_peak_memory_stats()
        embeddings = encode_rds_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            batch_size=int(contract["representation"]["batch_size"]),
            max_length=int(contract["representation"]["max_length"]),
            batch_callback=thermal_callback,
        )
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        peak_memory_gib = peak_memory_bytes / 1024**3
        if peak_memory_gib > float(thermal["maximum_peak_memory_gib"]):
            raise RuntimeError(
                f"peak GPU allocation {peak_memory_gib:.3f} GiB exceeds limit"
            )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if embeddings is None:
        raise RuntimeError("embedding computation did not produce a tensor")
    elapsed = time.perf_counter() - started
    gpu_end = _nvidia_snapshot()
    gpu_end["event"] = "after_gpu_cleanup"
    samples.append(gpu_end)
    expected_ids = [str(row[id_field]) for row in rows]
    attempt_id = (
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:12]}"
    )
    artifact_name = f"attempt_{attempt_id}.pt"
    artifact_path = directory / artifact_name
    payload = {
        "schema_version": CHUNK_SCHEMA,
        "kind": args.kind,
        "chunk_index": int(args.chunk_index),
        "run_contract_sha256": contract["run_contract_sha256"],
        "representation_version": RDS_FORMAT_VERSION,
        "ids": expected_ids,
        "embeddings": embeddings,
    }
    with artifact_path.open("xb") as handle:
        torch.save(payload, handle)
    manifest = {
        "schema_version": CHUNK_SCHEMA,
        "status": "COMPLETE",
        "kind": args.kind,
        "chunk_index": int(args.chunk_index),
        "start_index": start,
        "end_index_exclusive": end,
        "row_count": len(rows),
        "ordered_id_sha256": ordered_value_sha256(expected_ids),
        "run_contract_sha256": contract["run_contract_sha256"],
        "representation_version": RDS_FORMAT_VERSION,
        "artifact_file": artifact_name,
        "artifact_sha256": file_sha256(artifact_path),
        "embedding_tensor_sha256": tensor_sha256(embeddings),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gib": peak_memory_gib,
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
        "thermal_samples": samples,
        "maximum_observed_temperature_c": max(
            int(sample["temperature_c"]) for sample in samples
        ),
        "system_start": system_start,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "source_git_commit": _git_commit(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "This artifact contains frozen RDS+ representations only. "
            "It is not a training or downstream evaluation result."
        ),
    }
    _write_json(fixed_manifest, manifest)
    print(json.dumps({"manifest": str(fixed_manifest), **manifest}, indent=2))


def _status_payload(
    *,
    run_dir: Path,
    contract: dict[str, Any],
    prepared: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
    deep: bool,
) -> dict[str, Any]:
    kinds: dict[str, Any] = {}
    for kind in ("query", "candidate"):
        inventory, id_field = _inventory_for_kind(
            kind=kind,
            candidates=candidates,
            queries=queries,
        )
        chunk_size = int(
            contract["representation"][
                "query_chunk_size"
                if kind == "query"
                else "candidate_chunk_size"
            ]
        )
        expected_count = chunk_count(len(inventory), chunk_size)
        complete: list[int] = []
        invalid: list[dict[str, Any]] = []
        for chunk_index in range(expected_count):
            start, end = chunk_bounds(len(inventory), chunk_size, chunk_index)
            manifest_path = (
                _chunk_directory(run_dir, kind)
                / chunk_manifest_filename(chunk_index)
            )
            if not manifest_path.is_file():
                continue
            try:
                _load_chunk(
                    run_dir=run_dir,
                    contract=contract,
                    kind=kind,
                    chunk_index=chunk_index,
                    expected_rows=inventory[start:end],
                    id_field=id_field,
                    deep=deep,
                )
                complete.append(chunk_index)
            except (KeyError, OSError, TypeError, ValueError) as error:
                invalid.append(
                    {
                        "chunk_index": chunk_index,
                        "error": str(error),
                    }
                )
        kinds[kind] = {
            "inventory_row_count": len(inventory),
            "chunk_size": chunk_size,
            "expected_chunk_count": expected_count,
            "complete_chunk_count": len(complete),
            "complete_chunk_indices": complete,
            "missing_chunk_indices": [
                index for index in range(expected_count) if index not in complete
            ],
            "invalid_chunks": invalid,
        }
    ready = all(
        not report["missing_chunk_indices"] and not report["invalid_chunks"]
        for report in kinds.values()
    )
    return {
        "run_dir": str(run_dir),
        "run_contract_sha256": contract["run_contract_sha256"],
        "scope": prepared["scope"],
        "ready_to_finalize": ready,
        "kinds": kinds,
    }


def _status(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    contract = _load_contract(run_dir)
    prepared, candidates, queries = _load_prepared(run_dir, contract)
    print(
        json.dumps(
            _status_payload(
                run_dir=run_dir,
                contract=contract,
                prepared=prepared,
                candidates=candidates,
                queries=queries,
                deep=args.deep,
            ),
            indent=2,
        )
    )


def _load_all_embeddings(
    *,
    run_dir: Path,
    contract: dict[str, Any],
    kind: str,
    inventory: Sequence[dict[str, Any]],
    id_field: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    chunk_size = int(
        contract["representation"][
            "query_chunk_size" if kind == "query" else "candidate_chunk_size"
        ]
    )
    tensors: list[torch.Tensor] = []
    manifests: list[dict[str, Any]] = []
    for chunk_index in range(chunk_count(len(inventory), chunk_size)):
        start, end = chunk_bounds(len(inventory), chunk_size, chunk_index)
        manifest, payload = _load_chunk(
            run_dir=run_dir,
            contract=contract,
            kind=kind,
            chunk_index=chunk_index,
            expected_rows=inventory[start:end],
            id_field=id_field,
            deep=True,
        )
        if payload is None:
            raise AssertionError("deep chunk validation returned no payload")
        tensors.append(payload["embeddings"])
        manifests.append(manifest)
    combined = torch.cat(tensors, dim=0)
    if combined.shape[0] != len(inventory):
        raise ValueError(f"combined {kind} embedding count changed")
    return combined, manifests


def _validate_existing_finalization(
    run_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    manifest = _read_json(run_dir / "finalization_manifest.json")
    if manifest.get("schema_version") != FINALIZATION_SCHEMA:
        raise ValueError("finalization schema changed")
    if manifest.get("run_contract_sha256") != contract["run_contract_sha256"]:
        raise ValueError("finalization belongs to another run contract")
    for artifact in manifest["artifacts"].values():
        path = run_dir / str(artifact["path"])
        if file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"finalized artifact changed: {path}")
    return manifest


def _finalize(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    contract = _load_contract(run_dir)
    prepared, candidates, queries = _load_prepared(run_dir, contract)
    fixed_manifest = run_dir / "finalization_manifest.json"
    if fixed_manifest.is_file():
        manifest = _validate_existing_finalization(run_dir, contract)
        print(json.dumps({"status": "ALREADY_COMPLETE", **manifest}, indent=2))
        return
    status = _status_payload(
        run_dir=run_dir,
        contract=contract,
        prepared=prepared,
        candidates=candidates,
        queries=queries,
        deep=True,
    )
    if not status["ready_to_finalize"]:
        raise RuntimeError("embedding chunks are incomplete or invalid")
    eligible = [
        row for row in candidates if row["response_only_trainable"]
    ]
    query_embeddings, query_manifests = _load_all_embeddings(
        run_dir=run_dir,
        contract=contract,
        kind="query",
        inventory=queries,
        id_field="record_id",
    )
    candidate_embeddings, candidate_manifests = _load_all_embeddings(
        run_dir=run_dir,
        contract=contract,
        kind="candidate",
        inventory=eligible,
        id_field="candidate_id",
    )
    frozen_budget = int(contract["selection"]["budget"])
    scoring_budget = min(frozen_budget, len(eligible))
    score_rows, score_metrics = build_score_rows(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        query_inventory=queries,
        eligible_candidates=eligible,
        selection_budget=scoring_budget,
    )
    attempt_id = (
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:12]}"
    )
    attempt_dir = run_dir / "finalization_attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    scores_path = attempt_dir / "candidate_scores.jsonl"
    metrics_path = attempt_dir / "metrics.json"
    _write_jsonl(scores_path, score_rows)
    metrics = {
        "status": "COMPLETE",
        "scope": prepared["scope"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "representation_version": RDS_FORMAT_VERSION,
        "audited_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "excluded_fully_truncated_count": len(candidates) - len(eligible),
        "all_query_count": len(queries),
        "error_query_count": sum(
            bool(row["is_error_query"]) for row in queries
        ),
        "query_embeddings_sha256": tensor_sha256(query_embeddings),
        "candidate_embeddings_sha256": tensor_sha256(candidate_embeddings),
        "candidate_scores_sha256": file_sha256(scores_path),
        "eligible_ordered_id_sha256": ordered_value_sha256(
            [str(row["candidate_id"]) for row in eligible]
        ),
        "eligible_source_counts": dict(
            sorted(Counter(str(row["source_dataset"]) for row in eligible).items())
        ),
        "eligible_total_tokens": sum(int(row["total_tokens"]) for row in eligible),
        "eligible_supervised_tokens": sum(
            int(row["supervised_tokens"]) for row in eligible
        ),
        "query_chunk_manifest_sha256": [
            canonical_json_sha256(manifest) for manifest in query_manifests
        ],
        "candidate_chunk_manifest_sha256": [
            canonical_json_sha256(manifest) for manifest in candidate_manifests
        ],
        **score_metrics,
        "claim_boundary": contract["claim_boundary"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(metrics_path, metrics)
    finalization = {
        "schema_version": FINALIZATION_SCHEMA,
        "status": "COMPLETE",
        "scope": prepared["scope"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "artifacts": {
            "candidate_scores": {
                "path": scores_path.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(scores_path),
                "row_count": len(score_rows),
            },
            "metrics": {
                "path": metrics_path.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(metrics_path),
            },
        },
        "finalization_attempt_id": attempt_id,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": contract["claim_boundary"],
    }
    _write_json(fixed_manifest, finalization)
    print(json.dumps({"manifest": str(fixed_manifest), **finalization}, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol-config", type=Path, required=True)
    prepare.add_argument("--execution-config", type=Path, required=True)
    prepare.add_argument("--data-manifest-dir", type=Path, required=True)
    prepare.add_argument("--query-groups-dir", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--smoke", action="store_true")
    prepare.add_argument("--candidate-limit", type=int)
    prepare.add_argument("--query-limit", type=int)
    prepare.set_defaults(function=_prepare)

    encode = subparsers.add_parser("encode")
    encode.add_argument("--run-dir", type=Path, required=True)
    encode.add_argument("--kind", choices=("query", "candidate"), required=True)
    encode.add_argument("--chunk-index", type=int, required=True)
    encode.set_defaults(function=_encode)

    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--deep", action="store_true")
    status.set_defaults(function=_status)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.set_defaults(function=_finalize)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
