"""Pure helpers for the frozen 16-item batch-shape falsification."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "batch-shape-falsification16-v1"
PASS_IDS = (
    "bf16_b1_natural_repeat",
    "bf16_b4_fixed_a",
    "bf16_b4_fixed_b",
    "bf16_b1_fixed",
    "fp32_b1_fixed",
    "fp32_b4_fixed",
)
PHASE_REQUIREMENTS = {
    "baseline_repeat": ("bf16_b1_natural_repeat",),
    "bf16_repeat": ("bf16_b4_fixed_a", "bf16_b4_fixed_b"),
    "width_effect": ("bf16_b1_natural_repeat", "bf16_b1_fixed"),
    "final_mechanism": (
        "bf16_b1_fixed",
        "bf16_b4_fixed_a",
        "fp32_b1_fixed",
        "fp32_b4_fixed",
    ),
}


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive_or_verify(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable artifact changed: {path.name}")
        return
    path.write_bytes(content)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_token_ids(token_ids: Sequence[int], eos_token_id: int) -> list[int]:
    result: list[int] = []
    for value in token_ids:
        token = int(value)
        result.append(token)
        if token == eos_token_id:
            break
    return result


def first_eos_index(token_ids: Sequence[int], eos_token_id: int) -> int | None:
    for index, value in enumerate(token_ids):
        if int(value) == eos_token_id:
            return index
    return None


def effective_token_count(token_ids: Sequence[int], eos_token_id: int) -> int:
    canonical = canonical_token_ids(token_ids, eos_token_id)
    if canonical and canonical[-1] == eos_token_id:
        return len(canonical) - 1
    return len(canonical)


def normalize_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    try:
        return str(Decimal(text).normalize())
    except InvalidOperation:
        return text


def semantic_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        normalize_number(row.get("parsed_prediction")),
        bool(row.get("numeric_correct")),
        row.get("strict_parse_status"),
        row.get("parse_mode"),
        row.get("parse_status"),
    )


def row_canonical_ids(row: Mapping[str, Any], eos_token_id: int) -> list[int]:
    if "canonical_generated_ids" in row:
        return [int(value) for value in row["canonical_generated_ids"]]
    return canonical_token_ids(row["generated_token_ids"], eos_token_id)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected falsification config schema")
    if tuple(config.get("phase_order", ())) != PASS_IDS:
        raise ValueError("six-pass order changed")
    passes = config.get("passes", ())
    if not isinstance(passes, list) or tuple(row.get("pass_id") for row in passes) != PASS_IDS:
        raise ValueError("pass specifications changed")
    selection = config.get("selection", {})
    persistent = selection.get("persistent_semantic_mismatch_ids", ())
    controls = selection.get("lowest_risk_control_ids", ())
    selected = [*persistent, *controls]
    if len(persistent) != 12 or len(controls) != 4:
        raise ValueError("selection strata must contain 12 persistent and 4 controls")
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("selection must contain 16 unique record IDs")
    gates = config.get("gates", {})
    if gates.get("exact_canonical_token_match_required") is not True:
        raise ValueError("canonical token gate cannot be relaxed")
    if gates.get("exact_l0_to_l4_match_required") is not True:
        raise ValueError("L0-L4 gate cannot be relaxed")
    if gates.get("batch_gt_1_authorized") is not False:
        raise ValueError("this diagnostic cannot authorize batch>1")
    if gates.get("accuracy_aggregation_forbidden") is not True:
        raise ValueError("accuracy aggregation must remain forbidden")
    generation = config.get("generation", {})
    expected_generation = {
        "do_sample": False,
        "num_beams": 1,
        "max_input_length": 512,
        "max_new_tokens": 256,
        "use_cache": True,
        "position_ids_policy": "implicit_transformers_generate_from_attention_mask",
        "explicit_position_ids_passed": False,
    }
    for field, expected in expected_generation.items():
        if generation.get(field) != expected:
            raise ValueError(f"generation field changed: {field}")
    tokenizer = config.get("tokenizer", {})
    if tokenizer.get("padding_side") != "left":
        raise ValueError("only left padding is permitted")
    if tokenizer.get("pad_token_id") != tokenizer.get("eos_token_id"):
        raise ValueError("frozen tokenizer pad/eos binding changed")


def source_run_dir(source_root: Path, batch_size: int) -> Path:
    return source_root / f"smoke128__base__b{batch_size}"


def validate_source_runs(
    *, source_root: Path, config: Mapping[str, Any]
) -> dict[int, list[dict[str, Any]]]:
    required = config["source_smoke"]["required_runs"]
    rows: dict[int, list[dict[str, Any]]] = {}
    for batch_size in (1, 2, 4, 8):
        directory = source_run_dir(source_root, batch_size)
        binding = required[str(batch_size)]
        paths = {
            "raw_outputs_sha256": directory / "raw_outputs.jsonl",
            "metrics_sha256": directory / "metrics.json",
            "manifest_sha256": directory / "manifest.json",
        }
        for field, path in paths.items():
            if not path.is_file() or file_sha256(path) != binding[field]:
                raise ValueError(f"source smoke binding changed: batch={batch_size}, {field}")
        batch_rows = read_jsonl(paths["raw_outputs_sha256"])
        if len(batch_rows) != 128:
            raise ValueError(f"source batch {batch_size} does not contain 128 rows")
        rows[batch_size] = batch_rows
    reference_ids = [str(row["record_id"]) for row in rows[1]]
    for batch_size in (2, 4, 8):
        if [str(row["record_id"]) for row in rows[batch_size]] != reference_ids:
            raise ValueError(f"source record order changed for batch {batch_size}")
    return rows


def derive_selection(
    *, source_rows: Mapping[int, Sequence[Mapping[str, Any]]], eos_token_id: int
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for index, reference in enumerate(source_rows[1]):
        semantic_mismatches = sum(
            semantic_signature(source_rows[batch][index])
            != semantic_signature(reference)
            for batch in (2, 4, 8)
        )
        canonical_mismatches = sum(
            row_canonical_ids(source_rows[batch][index], eos_token_id)
            != row_canonical_ids(reference, eos_token_id)
            for batch in (2, 4, 8)
        )
        raw_mismatches = sum(
            list(source_rows[batch][index]["generated_token_ids"])
            != list(reference["generated_token_ids"])
            for batch in (2, 4, 8)
        )
        ranked.append(
            {
                "record_id": str(reference["record_id"]),
                "source_index": int(reference["source_index"]),
                "semantic_mismatch_frequency": semantic_mismatches,
                "canonical_mismatch_frequency": canonical_mismatches,
                "raw_mismatch_frequency": raw_mismatches,
            }
        )
    persistent = sorted(
        (row for row in ranked if row["semantic_mismatch_frequency"] == 3),
        key=lambda row: row["record_id"],
    )
    if len(persistent) != 12:
        raise ValueError("expected exactly 12 persistent semantic mismatches")
    persistent_ids = {row["record_id"] for row in persistent}
    controls = sorted(
        (row for row in ranked if row["record_id"] not in persistent_ids),
        key=lambda row: (
            row["semantic_mismatch_frequency"],
            row["canonical_mismatch_frequency"],
            row["raw_mismatch_frequency"],
            row["record_id"],
        ),
    )[:4]
    selected = [*persistent, *controls]
    return {
        "schema_version": "batch-shape-falsification16-selection-v1",
        "rule": "all_12_persistent_semantic_mismatch_then_4_lowest_risk_controls_v1",
        "selected_count": len(selected),
        "selected_record_ids": [row["record_id"] for row in selected],
        "strata": {
            "persistent_semantic_mismatch": persistent,
            "lowest_risk_control": controls,
        },
    }


def validate_selection_against_config(
    *, selection: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    expected = [
        *config["selection"]["persistent_semantic_mismatch_ids"],
        *config["selection"]["lowest_risk_control_ids"],
    ]
    if list(selection.get("selected_record_ids", ())) != expected:
        raise ValueError("derived 16-item selection differs from frozen config")


def compare_rows(
    *,
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    eos_token_id: int,
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("comparison record count changed")
    counts = {
        "l0_canonical_token_mismatch": 0,
        "l1_text_mismatch": 0,
        "l2_normalized_numeric_mismatch": 0,
        "l3_correctness_mismatch": 0,
        "l4_format_or_parser_mismatch": 0,
    }
    mismatch_ids: dict[str, list[str]] = {key: [] for key in counts}
    for left, right in zip(reference, candidate, strict=True):
        record_id = str(left.get("record_id"))
        if record_id != str(right.get("record_id")):
            raise ValueError("comparison record order changed")
        checks = {
            "l0_canonical_token_mismatch": (
                row_canonical_ids(left, eos_token_id)
                != row_canonical_ids(right, eos_token_id)
            ),
            "l1_text_mismatch": left.get("raw_output") != right.get("raw_output"),
            "l2_normalized_numeric_mismatch": (
                normalize_number(left.get("parsed_prediction"))
                != normalize_number(right.get("parsed_prediction"))
            ),
            "l3_correctness_mismatch": (
                bool(left.get("numeric_correct")) != bool(right.get("numeric_correct"))
            ),
            "l4_format_or_parser_mismatch": any(
                left.get(field) != right.get(field)
                for field in (
                    "strict_parse_status",
                    "parse_mode",
                    "parse_status",
                )
            ),
        }
        for field, differs in checks.items():
            if differs:
                counts[field] += 1
                mismatch_ids[field].append(record_id)
    return {
        "record_count": len(reference),
        **counts,
        "mismatch_record_ids": mismatch_ids,
        "exact_l0_to_l4": all(value == 0 for value in counts.values()),
    }


def load_pass_rows(output_root: Path, pass_id: str) -> list[dict[str, Any]]:
    path = output_root / "runs" / pass_id / "raw_outputs.jsonl"
    if not path.is_file():
        raise ValueError(f"pass output missing: {pass_id}")
    rows = read_jsonl(path)
    if len(rows) != 16:
        raise ValueError(f"pass output incomplete: {pass_id}")
    return rows


def selected_source_reference(
    *, source_rows: Mapping[int, Sequence[Mapping[str, Any]]], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_id = {str(row["record_id"]): dict(row) for row in source_rows[1]}
    return [by_id[record_id] for record_id in selection["selected_record_ids"]]


def audit_phase(
    *,
    phase: str,
    output_root: Path,
    source_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if phase not in PHASE_REQUIREMENTS:
        raise ValueError(f"unknown audit phase: {phase}")
    eos = int(config["tokenizer"]["eos_token_id"])
    source = selected_source_reference(source_rows=source_rows, selection=selection)
    comparisons: dict[str, Any] = {}
    if phase == "baseline_repeat":
        comparisons["source_b1_vs_bf16_b1_natural_repeat"] = compare_rows(
            reference=source,
            candidate=load_pass_rows(output_root, "bf16_b1_natural_repeat"),
            eos_token_id=eos,
        )
    elif phase == "bf16_repeat":
        comparisons["bf16_b4_fixed_a_vs_b"] = compare_rows(
            reference=load_pass_rows(output_root, "bf16_b4_fixed_a"),
            candidate=load_pass_rows(output_root, "bf16_b4_fixed_b"),
            eos_token_id=eos,
        )
    elif phase == "width_effect":
        fixed = load_pass_rows(output_root, "bf16_b1_fixed")
        comparisons["source_b1_vs_bf16_b1_fixed"] = compare_rows(
            reference=source, candidate=fixed, eos_token_id=eos
        )
        comparisons["natural_repeat_vs_bf16_b1_fixed"] = compare_rows(
            reference=load_pass_rows(output_root, "bf16_b1_natural_repeat"),
            candidate=fixed,
            eos_token_id=eos,
        )
    else:
        comparisons["bf16_b1_fixed_vs_b4_fixed"] = compare_rows(
            reference=load_pass_rows(output_root, "bf16_b1_fixed"),
            candidate=load_pass_rows(output_root, "bf16_b4_fixed_a"),
            eos_token_id=eos,
        )
        comparisons["fp32_b1_fixed_vs_b4_fixed"] = compare_rows(
            reference=load_pass_rows(output_root, "fp32_b1_fixed"),
            candidate=load_pass_rows(output_root, "fp32_b4_fixed"),
            eos_token_id=eos,
        )
    all_exact = all(row["exact_l0_to_l4"] for row in comparisons.values())
    if phase != "final_mechanism":
        decision = "CONTINUE" if all_exact else "PERMANENT_BATCH1"
    else:
        bf16 = comparisons["bf16_b1_fixed_vs_b4_fixed"]
        fp32 = comparisons["fp32_b1_fixed_vs_b4_fixed"]
        if bf16["exact_l0_to_l4"]:
            decision = "LIMITED_REQUALIFICATION_CANDIDATE"
        elif (
            fp32["l0_canonical_token_mismatch"]
            <= int(config["gates"]["fp32_near_equivalence_max_mismatch"])
        ):
            decision = "PERMANENT_BATCH1_PRECISION_SHAPE_CONFIRMED"
        elif (
            fp32["l0_canonical_token_mismatch"]
            >= int(config["gates"]["limited_debug_min_fp32_mismatch"])
        ):
            decision = "LIMITED_DEBUG_ALLOWED"
        else:
            decision = "PERMANENT_BATCH1_INCONCLUSIVE"
    return {
        "schema_version": "batch-shape-falsification16-audit-v1",
        "phase": phase,
        "status": "PASS" if all_exact else "FAIL",
        "decision": decision,
        "comparisons": comparisons,
        "batch_gt_1_authorized": False,
        "accuracy_aggregated": False,
        "gpu_accessed_by_audit": False,
    }
