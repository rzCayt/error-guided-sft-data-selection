from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_residual_selector_identifiability as residual  # noqa: E402
import run_model_aware_f0_f1 as f01  # noqa: E402


STAGE_ID = "model_aware_signal_f2_tiny_feasibility"
SEED = 20260711
SEED_TEXT = str(SEED)
CANDIDATE_QUOTAS = {"easy": 3, "medium": 3, "hard": 2}
PERMUTATION_COUNT = 1000
ID_TOLERANCE = 1e-7
ORDER_TOLERANCE = 1e-6
ALLOCATED_LIMIT_MIB = 7168.0
RESERVED_LIMIT_MIB = 7680.0
PREFLIGHT_FREE_MIB = 7680.0
MAX_MEDIAN_BACKWARD_SECONDS = 120.0

DEFAULT_OUTPUT_DIR = ROOT / "results/model_aware_signal_f2"
F0_DIR = ROOT / "results/model_aware_signal_f0_f1"
STATIC_PATH = F0_DIR / "static_estimate.json"
MEMORY_PATH = F0_DIR / "memory_smoke.json"
CORRECT_AUDIT_PATH = F0_DIR / "correct_query_length_audit.json"
F0_REVIEW_PATH = ROOT / "workflow/packets/model_aware_signal_f0_f1_review_response.json"

EXPECTED_ERROR_COUNT = 17
EXPECTED_CORRECT_COUNT = 8
EXPECTED_CANDIDATE_COUNT = 8


class HardStop(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def prepare_output_dir(output_dir: Path) -> None:
    protected = (
        "summary.json",
        "candidate_scores.csv",
        "permutation_null.json",
        "loo_stability.json",
        "run_metadata.json",
    )
    existing = [name for name in protected if (output_dir / name).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite F2 artifacts: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)


def selection_hash(row: dict[str, Any]) -> str:
    payload = (SEED_TEXT + f01.content_hash(row)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def select_f2_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for difficulty in ("easy", "medium", "hard"):
        stratum = [row for row in candidates if row.get("difficulty") == difficulty]
        quota = CANDIDATE_QUOTAS[difficulty]
        if len(stratum) < quota:
            raise ValueError(f"insufficient {difficulty} candidates: {len(stratum)}")
        selected.extend(sorted(stratum, key=selection_hash)[:quota])
    if len(selected) != EXPECTED_CANDIDATE_COUNT:
        raise AssertionError("candidate quota did not produce exactly eight rows")
    if len({f01.content_hash(row) for row in selected}) != len(selected):
        raise ValueError("selected candidates contain duplicate content")
    return selected


def verify_id_rename_sampling_invariance(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    original = {f01.content_hash(row) for row in select_f2_candidates(candidates)}
    renamed = [{**row, "id": f"renamed-{index:04d}"} for index, row in enumerate(candidates)]
    renamed_selection = {f01.content_hash(row) for row in select_f2_candidates(renamed)}
    return {
        "passed": original == renamed_selection,
        "original_content_hashes": sorted(original),
        "renamed_content_hashes": sorted(renamed_selection),
        "max_abs_score_difference": 0.0,
        "tolerance": ID_TOLERANCE,
        "scope": "candidate sampling and content-keyed offline scoring; IDs are not model inputs",
    }


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors must be non-empty and equally sized")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else 0.0


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(_ranks(left), _ranks(right))


def empirical_percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 < probability <= 1.0:
        raise ValueError("invalid empirical percentile request")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _prototype_scores(
    candidate_query_cosines: list[list[float]],
    query_gram: list[list[float]],
    query_indices: tuple[int, ...],
) -> list[float]:
    norm_squared = sum(query_gram[left][right] for left in query_indices for right in query_indices)
    if not math.isfinite(norm_squared) or norm_squared <= 1e-12:
        raise HardStop("prototype_norm_below_threshold")
    norm = math.sqrt(norm_squared)
    return [sum(row[index] for index in query_indices) / norm for row in candidate_query_cosines]


def compute_group_scores(
    candidate_query_cosines: list[list[float]],
    query_gram: list[list[float]],
    error_indices: tuple[int, ...],
    correct_indices: tuple[int, ...],
) -> dict[str, Any]:
    s_e = _prototype_scores(candidate_query_cosines, query_gram, error_indices)
    s_c = _prototype_scores(candidate_query_cosines, query_gram, correct_indices)
    delta = [left - right for left, right in zip(s_e, s_c)]
    rho = spearman(s_e, s_c)
    return {
        "s_e": s_e,
        "s_c": s_c,
        "delta": delta,
        "spearman_s_e_s_c": rho,
        "t_rank": 1.0 - rho,
        "t_delta": math.sqrt(sum(value * value for value in delta) / len(delta)),
    }


def top_k_indices(values: list[float], content_hashes: list[str], k: int) -> set[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], content_hashes[index]))
    return set(ordered[:k])


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def run_permutation_analysis(
    candidate_query_cosines: list[list[float]],
    query_gram: list[list[float]],
) -> dict[str, Any]:
    observed_errors = tuple(range(EXPECTED_ERROR_COUNT))
    all_indices = tuple(range(EXPECTED_ERROR_COUNT + EXPECTED_CORRECT_COUNT))
    rng = random.Random(SEED)
    sampled: set[tuple[int, ...]] = set()
    records: list[dict[str, Any]] = []
    while len(records) < PERMUTATION_COUNT:
        pseudo_errors = tuple(sorted(rng.sample(all_indices, EXPECTED_ERROR_COUNT)))
        if pseudo_errors == observed_errors or pseudo_errors in sampled:
            continue
        sampled.add(pseudo_errors)
        error_set = set(pseudo_errors)
        pseudo_correct = tuple(index for index in all_indices if index not in error_set)
        scores = compute_group_scores(
            candidate_query_cosines, query_gram, pseudo_errors, pseudo_correct
        )
        records.append(
            {
                "permutation_index": len(records),
                "pseudo_error_indices": list(pseudo_errors),
                "t_rank": scores["t_rank"],
                "t_delta": scores["t_delta"],
            }
        )
    t_rank_values = [record["t_rank"] for record in records]
    t_delta_values = [record["t_delta"] for record in records]
    return {
        "seed": SEED,
        "permutation_count": PERMUTATION_COUNT,
        "observed_assignment_excluded": True,
        "unique_assignments": len(sampled) == PERMUTATION_COUNT,
        "percentile_method": "empirical_nearest_rank",
        "p90_t_rank": empirical_percentile(t_rank_values, 0.90),
        "p90_t_delta": empirical_percentile(t_delta_values, 0.90),
        "records": records,
    }


def run_loo_analysis(
    candidate_query_cosines: list[list[float]],
    query_gram: list[list[float]],
    full_s_e: list[float],
    candidate_hashes: list[str],
    error_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    full_errors = tuple(range(EXPECTED_ERROR_COUNT))
    full_top = top_k_indices(full_s_e, candidate_hashes, 2)
    records: list[dict[str, Any]] = []
    for removed_index, row in enumerate(error_rows):
        kept = tuple(index for index in full_errors if index != removed_index)
        loo_s_e = _prototype_scores(candidate_query_cosines, query_gram, kept)
        records.append(
            {
                "removed_error_index": removed_index,
                "removed_error_id": row["id"],
                "spearman_vs_full": spearman(full_s_e, loo_s_e),
                "top_k": 2,
                "top_k_jaccard_vs_full": jaccard(
                    full_top, top_k_indices(loo_s_e, candidate_hashes, 2)
                ),
            }
        )
    correlations = sorted(record["spearman_vs_full"] for record in records)
    jaccards = [record["top_k_jaccard_vs_full"] for record in records]
    return {
        "leave_one_out_count": len(records),
        "top_k": 2,
        "spearman_median": median(correlations),
        "spearman_p10": empirical_percentile(correlations, 0.10),
        "top_k_jaccard_median": median(jaccards),
        "records": records,
    }


def normalize_prompt_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[-+]?\d+(?:\.\d+)?", " <num> ", normalized)
    return re.findall(r"[a-z]+|<num>", normalized)


def bm25_prompt_only_scores(
    candidates: list[dict[str, Any]], errors: list[dict[str, Any]]
) -> list[float]:
    documents = [normalize_prompt_tokens(row["prompt"]) for row in candidates]
    queries = [normalize_prompt_tokens(row["prompt"]) for row in errors]
    document_frequency = Counter(token for document in documents for token in set(document))
    average_length = sum(map(len, documents)) / len(documents)
    count = len(documents)

    def score(query: list[str], document: list[str]) -> float:
        frequencies = Counter(document)
        total = 0.0
        for token in set(query):
            frequency = frequencies[token]
            if not frequency:
                continue
            df = document_frequency[token]
            idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * len(document) / average_length
            )
            total += idf * frequency * 2.5 / denominator
        return total

    return [max(score(query, document) for query in queries) for document in documents]


def static_operation_scores(
    candidates: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    correct: list[dict[str, Any]],
) -> tuple[list[float], list[str]]:
    error_features = [residual.operation_features(row) for row in errors]
    correct_features = [residual.operation_features(row) for row in correct]

    def similarity(left: frozenset[str], right: frozenset[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 1.0

    def top_three(values: list[float]) -> float:
        chosen = sorted(values, reverse=True)[: min(3, len(values))]
        return sum(chosen) / len(chosen)

    scores: list[float] = []
    signatures: list[str] = []
    for row in candidates:
        features = residual.operation_features(row)
        scores.append(
            top_three([similarity(features, item) for item in error_features])
            - top_three([similarity(features, item) for item in correct_features])
        )
        signatures.append("|".join(sorted(features)))
    return scores, signatures


def exact_stratum_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    buckets = row["buckets"]
    return (
        row["task_family"],
        buckets["difficulty_bucket"],
        buckets["answer_magnitude_bucket"],
        buckets["reasoning_length_bucket"],
    )


def summarize_confound_correlations(
    rows: list[dict[str, Any]], s_e: list[float]
) -> dict[str, Any]:
    fields = {
        "token_normalized_loss": [float(row["token_normalized_loss"]) for row in rows],
        "gradient_norm": [float(row["gradient_norm"]) for row in rows],
        "target_token_count": [float(row["target_token_count"]) for row in rows],
        "final_answer_token_count": [float(row["final_answer_token_count"]) for row in rows],
        "answer_log1p_abs": [math.log1p(abs(float(row["answer"]))) for row in rows],
        "difficulty_ordinal": [float({"easy": 0, "medium": 1, "hard": 2}[row["difficulty"]]) for row in rows],
        "bm25_prompt_only": [float(row["bm25_prompt_only"]) for row in rows],
        "static_operation": [float(row["static_operation_score"]) for row in rows],
    }
    return {
        name: {
            "spearman_with_s_e": spearman(s_e, values),
            "unique_value_count": len(set(values)),
        }
        for name, values in fields.items()
    }


def stratum_variation(rows: list[dict[str, Any]], s_e: list[float]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row, score in zip(rows, s_e):
        groups[exact_stratum_key(row)].append(score)
    multi = [values for values in groups.values() if len(values) > 1]
    return {
        "exact_stratum_count": len(groups),
        "multi_candidate_exact_stratum_count": len(multi),
        "multi_candidate_nonconstant_count": sum(len(set(values)) > 1 for values in multi),
        "maximum_unique_s_e_within_exact_stratum": max(
            (len(set(values)) for values in multi), default=0
        ),
    }


def verify_prerequisites() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path in (STATIC_PATH, MEMORY_PATH, CORRECT_AUDIT_PATH, F0_REVIEW_PATH):
        if not path.exists():
            raise HardStop(f"missing_prerequisite:{path.relative_to(ROOT)}")
        lowered = str(path).lower()
        if any(marker in lowered for marker in ("test_id", "test_ood", "pair_manifest", "hidden_group")):
            raise HardStop(f"forbidden_prerequisite_path:{path}")
    static = read_json(STATIC_PATH)
    memory = read_json(MEMORY_PATH)
    correct_audit = read_json(CORRECT_AUDIT_PATH)
    review = read_json(F0_REVIEW_PATH)
    if static.get("status") != "passed" or memory.get("status") != "passed":
        raise HardStop("f0_f1_not_passed")
    if correct_audit.get("status") != "passed":
        raise HardStop("correct_query_length_audit_not_passed")
    decision = review.get("阶段判定", {})
    if not decision.get("allow_next_stage") or decision.get("allowed_next_stage") != STAGE_ID:
        raise HardStop("f0_f1_review_did_not_authorize_f2")
    if static.get("model") != f01.MODEL or static.get("revision") != f01.REVISION:
        raise HardStop("f0_model_or_revision_mismatch")
    return static, memory, correct_audit, review


def extract_normalized_gradient(
    torch: Any,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    kind: str,
    selected_parameter_names: list[str],
) -> tuple[dict[str, Any], Any]:
    serialized = f01.serialize_teacher_forcing(tokenizer, row)
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    input_ids = torch.tensor([serialized["input_ids"]], dtype=torch.long, device="cuda")
    attention_mask = torch.tensor(
        [serialized["attention_mask"]], dtype=torch.long, device="cuda"
    )
    labels = torch.tensor([serialized["labels"]], dtype=torch.long, device="cuda")
    started = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss
    if not bool(torch.isfinite(loss).item()):
        raise HardStop(f"non_finite_loss:{row['id']}")
    loss.backward()
    torch.cuda.synchronize()

    named_parameters = dict(model.named_parameters())
    gradients = [named_parameters[name].grad for name in selected_parameter_names]
    if any(gradient is None for gradient in gradients):
        raise HardStop(f"missing_selected_gradient:{row['id']}")
    if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
        raise HardStop(f"non_finite_gradient:{row['id']}")
    unexpected = [
        name
        for name, parameter in named_parameters.items()
        if name not in selected_parameter_names and parameter.grad is not None
    ]
    if unexpected:
        raise HardStop(f"unexpected_parameter_gradient:{row['id']}:{unexpected}")
    pieces = [gradient.detach().float().reshape(-1).cpu() for gradient in gradients]
    vector = torch.cat(pieces)
    gradient_norm = float(torch.linalg.vector_norm(vector).item())
    if not math.isfinite(gradient_norm) or gradient_norm < f01.MIN_GRADIENT_NORM:
        raise HardStop(f"gradient_norm_below_threshold:{row['id']}")
    vector.div_(gradient_norm + 1e-12)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = f01._mib(torch.cuda.max_memory_allocated())
    peak_reserved = f01._mib(torch.cuda.max_memory_reserved())
    if peak_allocated > ALLOCATED_LIMIT_MIB:
        raise HardStop(f"peak_allocated_limit_exceeded:{row['id']}:{peak_allocated}")
    if peak_reserved > RESERVED_LIMIT_MIB:
        raise HardStop(f"peak_reserved_limit_exceeded:{row['id']}:{peak_reserved}")

    loss_value = float(loss.detach().float().item())
    active_target_count = sum(label != -100 for label in serialized["labels"])
    if active_target_count != serialized["target_token_count"]:
        raise HardStop(f"assistant_mask_count_mismatch:{row['id']}")
    answer_tokens = tokenizer.encode(f01.canonical_answer(row["answer"]), add_special_tokens=False)
    rationale_tokens = tokenizer.encode(row["rationale"], add_special_tokens=False)
    record = {
        "kind": kind,
        "id": row["id"],
        "content_sha256": serialized["content_sha256"],
        "prefix_token_count": serialized["prefix_token_count"],
        "target_token_count": serialized["target_token_count"],
        "total_token_count": serialized["total_token_count"],
        "rationale_character_count": len(row["rationale"]),
        "rationale_token_count": len(rationale_tokens),
        "final_answer_token_count": len(answer_tokens),
        "token_normalized_loss": loss_value,
        "raw_summed_nll": loss_value * active_target_count,
        "perplexity": math.exp(loss_value) if loss_value < 700 else None,
        "gradient_norm": gradient_norm,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "elapsed_seconds": elapsed,
    }

    del pieces, outputs, loss, input_ids, attention_mask, labels
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return record, vector


def run_experiment(output_dir: Path) -> None:
    prepare_output_dir(output_dir)
    static, _, _, _ = verify_prerequisites()
    preflight = f01.query_nvidia_smi_memory_mib()
    if preflight["free"] < PREFLIGHT_FREE_MIB:
        raise HardStop(f"external_gpu_contention:free_mib={preflight['free']}")

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"F2 dependencies unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise HardStop("cuda_unavailable")

    all_candidates, error_rows, correct_rows = f01.load_frozen_rows()
    candidate_rows = select_f2_candidates(all_candidates)
    rename_invariance = verify_id_rename_sampling_invariance(all_candidates)
    if not rename_invariance["passed"]:
        raise HardStop("candidate_id_rename_invariance_failed")
    if len(error_rows) != EXPECTED_ERROR_COUNT or len(correct_rows) != EXPECTED_CORRECT_COUNT:
        raise HardStop("frozen_query_count_mismatch")

    tokenizer = AutoTokenizer.from_pretrained(f01.MODEL, revision=f01.REVISION)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            f01.MODEL,
            revision=f01.REVISION,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except TypeError:  # pragma: no cover - transformers compatibility
        model = AutoModelForCausalLM.from_pretrained(
            f01.MODEL,
            revision=f01.REVISION,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    model.to("cuda")
    model.eval()
    model.config.use_cache = False
    if getattr(model.config, "_commit_hash", None) != f01.REVISION:
        raise HardStop("loaded_model_revision_mismatch")
    if model.dtype != torch.bfloat16:
        raise HardStop(f"loaded_model_dtype_mismatch:{model.dtype}")

    selected_parameter_names = [
        str(item["name"]) for item in static["parameter_spec"]["parameters"]
    ]
    f01.configure_selected_parameters(model, selected_parameter_names)
    vector_size = int(static["parameter_spec"]["total_parameter_count"])
    candidate_matrix = torch.empty(
        (EXPECTED_CANDIDATE_COUNT, vector_size), dtype=torch.float32, device="cpu"
    )
    error_matrix = torch.empty(
        (EXPECTED_ERROR_COUNT, vector_size), dtype=torch.float32, device="cpu"
    )
    correct_matrix = torch.empty(
        (EXPECTED_CORRECT_COUNT, vector_size), dtype=torch.float32, device="cpu"
    )

    sample_records: list[dict[str, Any]] = []
    run_groups = (
        ("candidate", candidate_rows, candidate_matrix),
        ("error_query", error_rows, error_matrix),
        ("correct_query", correct_rows, correct_matrix),
    )
    for kind, rows, matrix in run_groups:
        for index, row in enumerate(rows):
            record, vector = extract_normalized_gradient(
                torch, model, tokenizer, row, kind, selected_parameter_names
            )
            matrix[index].copy_(vector)
            sample_records.append(record)
            del vector

    backward_times = [float(record["elapsed_seconds"]) for record in sample_records]
    if median(backward_times) > MAX_MEDIAN_BACKWARD_SECONDS:
        raise HardStop(f"median_backward_time_exceeded:{median(backward_times)}")

    pairwise_started = time.perf_counter()
    candidate_error = candidate_matrix @ error_matrix.T
    candidate_correct = candidate_matrix @ correct_matrix.T
    error_error = error_matrix @ error_matrix.T
    error_correct = error_matrix @ correct_matrix.T
    correct_correct = correct_matrix @ correct_matrix.T
    candidate_query_cosines = torch.cat((candidate_error, candidate_correct), dim=1).tolist()
    query_gram_tensor = torch.cat(
        (
            torch.cat((error_error, error_correct), dim=1),
            torch.cat((error_correct.T, correct_correct), dim=1),
        ),
        dim=0,
    )
    query_gram = query_gram_tensor.tolist()
    pairwise_seconds = time.perf_counter() - pairwise_started

    del (
        candidate_matrix,
        error_matrix,
        correct_matrix,
        candidate_error,
        candidate_correct,
        error_error,
        error_correct,
        correct_correct,
        query_gram_tensor,
        model,
    )
    gc.collect()
    torch.cuda.empty_cache()

    error_indices = tuple(range(EXPECTED_ERROR_COUNT))
    correct_indices = tuple(range(EXPECTED_ERROR_COUNT, EXPECTED_ERROR_COUNT + EXPECTED_CORRECT_COUNT))
    observed = compute_group_scores(
        candidate_query_cosines, query_gram, error_indices, correct_indices
    )
    permutation = run_permutation_analysis(candidate_query_cosines, query_gram)
    candidate_hashes = [f01.content_hash(row) for row in candidate_rows]
    loo = run_loo_analysis(
        candidate_query_cosines,
        query_gram,
        observed["s_e"],
        candidate_hashes,
        error_rows,
    )

    reversed_cross = [
        [row[index] for index in tuple(reversed(error_indices)) + tuple(reversed(correct_indices))]
        for row in reversed(candidate_query_cosines)
    ]
    reordered_query_indices = tuple(reversed(error_indices)) + tuple(reversed(correct_indices))
    reversed_gram = [
        [query_gram[left][right] for right in reordered_query_indices]
        for left in reordered_query_indices
    ]
    reversed_scores = compute_group_scores(
        reversed_cross,
        reversed_gram,
        tuple(range(EXPECTED_ERROR_COUNT)),
        tuple(range(EXPECTED_ERROR_COUNT, EXPECTED_ERROR_COUNT + EXPECTED_CORRECT_COUNT)),
    )
    restored_s_e = list(reversed(reversed_scores["s_e"]))
    restored_s_c = list(reversed(reversed_scores["s_c"]))
    order_max_difference = max(
        max(abs(left - right) for left, right in zip(observed["s_e"], restored_s_e)),
        max(abs(left - right) for left, right in zip(observed["s_c"], restored_s_c)),
    )
    order_invariance = {
        "passed": order_max_difference <= ORDER_TOLERANCE,
        "max_abs_score_difference": order_max_difference,
        "tolerance": ORDER_TOLERANCE,
        "scope": "offline aggregation after reversing candidate and within-group query order",
    }

    bm25_scores = bm25_prompt_only_scores(candidate_rows, error_rows)
    operation_scores, operation_signatures = static_operation_scores(
        candidate_rows, error_rows, correct_rows
    )
    candidate_run_records = [record for record in sample_records if record["kind"] == "candidate"]
    candidate_output_rows: list[dict[str, Any]] = []
    for index, (row, run_record) in enumerate(zip(candidate_rows, candidate_run_records)):
        params = row["metadata"]["params"]
        candidate_output_rows.append(
            {
                "candidate_id": row["id"],
                "content_sha256": candidate_hashes[index],
                "sampling_sha256": selection_hash(row),
                "difficulty": row["difficulty"],
                "answer_magnitude_bucket": row["buckets"]["answer_magnitude_bucket"],
                "reasoning_length_bucket": row["buckets"]["reasoning_length_bucket"],
                "term_count": len(params["values"]),
                "answer": row["answer"],
                "target_token_count": run_record["target_token_count"],
                "total_token_count": run_record["total_token_count"],
                "rationale_character_count": run_record["rationale_character_count"],
                "rationale_token_count": run_record["rationale_token_count"],
                "final_answer_token_count": run_record["final_answer_token_count"],
                "raw_summed_nll": run_record["raw_summed_nll"],
                "token_normalized_loss": run_record["token_normalized_loss"],
                "perplexity": run_record["perplexity"],
                "gradient_norm": run_record["gradient_norm"],
                "s_e": observed["s_e"][index],
                "s_c": observed["s_c"][index],
                "delta_e_minus_c": observed["delta"][index],
                "bm25_prompt_only": bm25_scores[index],
                "static_operation_score": operation_scores[index],
                "operation_signature": operation_signatures[index],
            }
        )

    correlations = summarize_confound_correlations(candidate_output_rows, observed["s_e"])
    variation = stratum_variation(candidate_rows, observed["s_e"])
    gates = {
        "spearman_s_e_s_c_le_0_90": observed["spearman_s_e_s_c"] <= 0.90,
        "t_rank_ge_permutation_p90": observed["t_rank"] >= permutation["p90_t_rank"],
        "t_delta_ge_permutation_p90": observed["t_delta"] >= permutation["p90_t_delta"],
        "loo_spearman_median_ge_0_80": loo["spearman_median"] >= 0.80,
        "loo_spearman_p10_ge_0_60": loo["spearman_p10"] >= 0.60,
        "loo_top_k_jaccard_median_ge_0_60": loo["top_k_jaccard_median"] >= 0.60,
        "id_rename_invariance": bool(rename_invariance["passed"]),
        "input_order_invariance": bool(order_invariance["passed"]),
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    status = "passed_minimum_feasibility_gate" if not failed_gates else "frozen_negative_result"

    summary = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "created_at_utc": utc_now(),
        "status": status,
        "model": f01.MODEL,
        "revision": f01.REVISION,
        "dtype": f01.DTYPE,
        "sample_counts": {
            "candidate": len(candidate_rows),
            "error_query": len(error_rows),
            "correct_query": len(correct_rows),
        },
        "observed": {
            "spearman_s_e_s_c": observed["spearman_s_e_s_c"],
            "t_rank": observed["t_rank"],
            "t_delta": observed["t_delta"],
        },
        "permutation_p90": {
            "t_rank": permutation["p90_t_rank"],
            "t_delta": permutation["p90_t_delta"],
        },
        "loo": {
            "spearman_median": loo["spearman_median"],
            "spearman_p10": loo["spearman_p10"],
            "top_k": loo["top_k"],
            "top_k_jaccard_median": loo["top_k_jaccard_median"],
        },
        "gates": gates,
        "failed_gates": failed_gates,
        "confound_correlations": correlations,
        "stratum_variation": variation,
        "invariance": {
            "candidate_id_rename": rename_invariance,
            "input_order": order_invariance,
        },
        "resource_summary": {
            "maximum_peak_allocated_mib": max(record["peak_allocated_mib"] for record in sample_records),
            "maximum_peak_reserved_mib": max(record["peak_reserved_mib"] for record in sample_records),
            "median_backward_seconds": median(backward_times),
            "maximum_backward_seconds": max(backward_times),
            "pairwise_analysis_seconds": pairwise_seconds,
        },
        "claim_boundary": (
            "This eight-candidate dev-only pilot tests representation feasibility only. "
            "It does not establish selector, training, or generalization effectiveness."
        ),
        "next_stage": None,
    }

    permutation_output = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "observed": {
            "t_rank": observed["t_rank"],
            "t_delta": observed["t_delta"],
        },
        **permutation,
    }
    loo_output = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        **loo,
    }
    run_metadata = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "created_at_utc": utc_now(),
        "status": status,
        "model": f01.MODEL,
        "revision": f01.REVISION,
        "dtype": f01.DTYPE,
        "device": {"type": "cuda", "name": torch.cuda.get_device_name(0)},
        "packages": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "preflight_nvidia_smi_memory_mib": preflight,
        "selected_parameter_names": selected_parameter_names,
        "parameter_count": vector_size,
        "candidate_sampling": {
            "seed_text": SEED_TEXT,
            "formula": "SHA256('20260711' + content_sha256), ascending within difficulty",
            "quotas": CANDIDATE_QUOTAS,
            "selected": [
                {
                    "id": row["id"],
                    "content_sha256": f01.content_hash(row),
                    "sampling_sha256": selection_hash(row),
                    "difficulty": row["difficulty"],
                }
                for row in candidate_rows
            ],
        },
        "input_hashes": {
            "candidate_pool_sha256": f01.sha256_file(f01.CANDIDATE_PATH),
            "dev_diagnostic_sha256": f01.sha256_file(f01.DEV_PATH),
            "error_membership_sha256": f01.sha256_file(f01.ERROR_MEMBERSHIP_PATH),
            "f0_static_estimate_sha256": f01.sha256_file(STATIC_PATH),
            "f0_memory_smoke_sha256": f01.sha256_file(MEMORY_PATH),
            "f0_correct_query_audit_sha256": f01.sha256_file(CORRECT_AUDIT_PATH),
            "f0_review_response_sha256": f01.sha256_file(F0_REVIEW_PATH),
        },
        "serialization": {
            "implementation": "run_model_aware_f0_f1.serialize_teacher_forcing",
            "assistant_only_mask": True,
            "max_sequence_tokens": f01.MAX_SEQUENCE_TOKENS,
            "truncation": False,
            "enable_thinking": False,
        },
        "gradient_storage": {
            "gpu_multi_sample_vectors": False,
            "cpu_normalized_fp32_transient": True,
            "raw_gradient_vectors_written": False,
            "saved_pairwise_candidate_query_cosines": False,
            "saved_query_gram": False,
        },
        "sample_records": sample_records,
        "analysis": {
            "permutation_count": PERMUTATION_COUNT,
            "permutation_seed": SEED,
            "loo_count": EXPECTED_ERROR_COUNT,
            "bm25": {
                "fields": ["prompt"],
                "unicode": "NFKC",
                "lowercase": True,
                "number_replacement": "<num>",
                "token_regex": "[a-z]+|<num>",
                "k1": 1.5,
                "b": 0.75,
                "aggregation": "maximum over 17 error queries",
            },
            "static_operation": "existing operation_features; top3(error Jaccard)-top3(correct Jaccard)",
            "input_order_invariance_scope": order_invariance["scope"],
        },
        "forbidden_inputs_used": [],
        "optimizer_created": False,
        "parameter_update_performed": False,
        "training_subset_written": False,
        "claim_boundary": summary["claim_boundary"],
    }

    write_json_atomic(output_dir / "summary.json", summary)
    write_csv_atomic(output_dir / "candidate_scores.csv", candidate_output_rows)
    write_json_atomic(output_dir / "permutation_null.json", permutation_output)
    write_json_atomic(output_dir / "loo_stability.json", loo_output)
    write_json_atomic(output_dir / "run_metadata.json", run_metadata)
    print(json.dumps({"status": status, "failed_gates": failed_gates}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen 8-candidate model-aware F2 representation feasibility pilot."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        run_experiment(args.output_dir.resolve())
    except HardStop as exc:
        print(f"F2 hard stop: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
