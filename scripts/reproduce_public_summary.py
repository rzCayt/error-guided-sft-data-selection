#!/usr/bin/env python3
"""Recompute the canonical public summary from audited source artifacts.

The command has two modes:

* ``--write`` creates the canonical JSON/CSV/Markdown files once.
* ``--check`` recomputes all payloads in memory and fails on any difference.

It never treats README prose or a previous summary as evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1]
PUBLIC_ROOT = ROOT / "results" / "public_summary"
SOURCES = {
    "phase2_primary": PUBLIC_ROOT
    / "evidence"
    / "phase2"
    / "final_analysis_v8_20260831"
    / "unblinded_analysis"
    / "accuracy_fixed_seed_primary.json",
    "phase2_sensitivity": PUBLIC_ROOT
    / "evidence"
    / "phase2"
    / "final_analysis_v8_20260831"
    / "unblinded_analysis"
    / "accuracy_seed_resampled_sensitivity.json",
    "phase2_claims": PUBLIC_ROOT
    / "evidence"
    / "phase2"
    / "final_analysis_v8_20260831"
    / "unblinded_analysis"
    / "claim_evidence_ledger.json",
    "h1a_tulu96": PUBLIC_ROOT / "evidence" / "h1a" / "tulu96_metrics.json",
    "h1a_domain48": PUBLIC_ROOT
    / "evidence"
    / "h1a"
    / "gsm8k_domain48_metrics.json",
    "cpu_composition": PUBLIC_ROOT
    / "evidence"
    / "cpu_composition"
    / "ARTIFACT_INDEX.json",
    "state_panel": ROOT / "artifacts" / "state_dependence_candidate_panel48_v3.json",
    "state_overlap": ROOT / "artifacts" / "state_dependence_overlap_audit_v3.json",
    "state_preflight": ROOT / "artifacts" / "state_dependence_preflight_v3.json",
}
OUTPUTS = {
    "json": PUBLIC_ROOT / "main_results.json",
    "csv": PUBLIC_ROOT / "main_results.csv",
    "markdown": PUBLIC_ROOT / "main_results_table.md",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def percent_points(value: float) -> float:
    return round(float(value) * 100.0, 3)


def build_payload() -> dict[str, Any]:
    source = {name: read_json(path) for name, path in SOURCES.items()}
    primary = source["phase2_primary"]
    sensitivity = source["phase2_sensitivity"]
    claims = source["phase2_claims"]
    tulu = source["h1a_tulu96"]
    domain = source["h1a_domain48"]
    composition = source["cpu_composition"]
    panel = source["state_panel"]
    overlap = source["state_overlap"]
    preflight = source["state_preflight"]

    require(
        primary.get("schema_version") == "phase2-v8-fixed-seed-block-bootstrap-v1",
        "unexpected Phase 2 primary schema",
    )
    require(primary.get("bootstrap_replicates") == 20000, "Phase 2 bootstrap count changed")
    require(
        sensitivity.get("schema_version") == "phase2-v8-seed-resampled-bootstrap-v1",
        "unexpected Phase 2 sensitivity schema",
    )
    require(claims.get("allowed_claim_class") == "insufficient_evidence", "claim class changed")
    require(tulu.get("candidate_count") == 96 and tulu.get("h1a_gate_passed") is True, "Tulu96 changed")
    require(
        domain.get("candidate_count") == 48 and domain.get("h1a_gate_passed") is False,
        "GSM8K-domain48 changed",
    )
    require(composition.get("status") == "NO_GO", "CPU composition verdict changed")
    require(panel.get("candidate_count") == 48, "state-dependence panel count changed")
    require(overlap.get("frozen_panel_overlap_count") == 0, "state panel has training overlap")
    require(
        preflight.get("status") == "READY_FOR_GPU_QUALIFICATION"
        and preflight.get("gpu_accessed") is False,
        "state-dependence preflight status changed",
    )

    endpoints: dict[str, dict[str, Any]] = {}
    for endpoint in ("gsm8k", "ood_macro", "svamp", "asdiv_numeric", "multiarith"):
        row = primary["results"][endpoint]
        endpoints[endpoint] = {
            "difference_fraction": float(row["point_difference"]),
            "difference_percentage_points": percent_points(row["point_difference"]),
            "ci95_fraction": [float(value) for value in row["ci95"]],
            "ci95_percentage_points": [percent_points(value) for value in row["ci95"]],
            "diagnostic": str(row["effect_diagnostic"]),
        }

    payload = {
        "schema_version": "public-research-summary-v1",
        "study_date": "2026-08-31",
        "research_question": (
            "Under matched sample count, response-supervision tokens, and data composition, "
            "does targeted instruction selection outperform matched random selection?"
        ),
        "completed_study": {
            "model": "Qwen/Qwen2.5-1.5B",
            "training_method": "LoRA response-only SFT",
            "methods": ["random_common_mix", "rds_error_common_mix"],
            "selection_list_count_per_method": 4,
            "training_seeds": [17, 29, 41],
            "cell_count": 24,
            "examples_per_list": 500,
            "primary_estimand": "rds_error_common_mix - random_common_mix",
            "primary_training_seed_role": "fixed_observed_blocks",
            "bootstrap_replicates": 20000,
        },
        "downstream_results": endpoints,
        "seed_resampled_sensitivity": {
            endpoint: {
                "difference_percentage_points": percent_points(row["point_difference"]),
                "ci95_percentage_points": [percent_points(value) for value in row["ci95"]],
                "diagnostic": str(row["effect_diagnostic"]),
            }
            for endpoint, row in sensitivity["results"].items()
        },
        "candidate_utility": {
            "tulu96": {
                "candidate_count": 96,
                "partial_spearman": float(tulu["observed_partial_spearman"]),
                "one_sided_permutation_p": float(tulu["one_sided_permutation_p"]),
                "top_minus_bottom_utility": float(tulu["top_minus_bottom_mean_utility"]),
                "original_gate_passed": True,
            },
            "gsm8k_domain48": {
                "candidate_count": 48,
                "partial_spearman": float(domain["observed_partial_spearman"]),
                "one_sided_permutation_p": float(domain["one_sided_permutation_p"]),
                "top_minus_bottom_utility": float(domain["top_minus_bottom_mean_utility"]),
                "original_gate_passed": False,
            },
        },
        "cpu_composition_audit": {
            "status": "NO_GO",
            "interpretation": (
                "No frozen response-composition feature passed all source-sensitivity and "
                "multiplicity gates; behavior-constrained retraining was not authorized."
            ),
        },
        "state_dependence_v3": {
            "status": "READY_FOR_GPU_QUALIFICATION",
            "gpu_result_available": False,
            "original_candidate_count": 96,
            "training_exposed_candidates_removed": int(overlap["score_panel_seen_count"]),
            "unseen_candidates_available": int(overlap["score_panel_unseen_count"]),
            "frozen_panel_count": int(panel["candidate_count"]),
            "training_overlap_count": int(overlap["frozen_panel_overlap_count"]),
            "selected_id_sha256": str(panel["selected_id_sha256"]),
            "u0a_planned_measurements": int(
                preflight["planned_new_measurements"]["u0a_fixed_state_reliability"]
            ),
            "u1_initial_planned_measurements": int(
                preflight["planned_new_measurements"][
                    "u1_initial_four_states_two_probe_seeds"
                ]
            ),
        },
        "claim_boundaries": {
            "supported": [
                "The 24-cell block found no reliable downstream advantage for the frozen RDS policy.",
                "Tulu96 passed its original candidate-utility gate; GSM8K-domain48 did not.",
                "The frozen response-composition mechanism did not pass its prespecified CPU gate.",
                "The State Dependence v3 panel is universal-unseen for the four initial adapters.",
            ],
            "not_supported": [
                "RDS is generally ineffective.",
                "RDS and Random are equivalent.",
                "State dependence has been observed.",
                "The final-adapter local probe reconstructs the historical optimizer trajectory.",
            ],
        },
        "evidence": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in SOURCES.items()
        },
    }
    return payload


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_csv(payload: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["endpoint", "difference_pp", "ci95_lower_pp", "ci95_upper_pp", "diagnostic"])
    for endpoint in ("gsm8k", "ood_macro", "svamp", "asdiv_numeric", "multiarith"):
        row = payload["downstream_results"][endpoint]
        writer.writerow(
            [
                endpoint,
                f'{row["difference_percentage_points"]:.3f}',
                f'{row["ci95_percentage_points"][0]:.3f}',
                f'{row["ci95_percentage_points"][1]:.3f}',
                row["diagnostic"],
            ]
        )
    return buffer.getvalue()


def render_markdown(payload: dict[str, Any]) -> str:
    labels = {
        "gsm8k": "GSM8K exact numeric",
        "ood_macro": "OOD arithmetic macro",
    }
    lines = [
        "<!-- Generated by scripts/reproduce_public_summary.py. Do not edit manually. -->",
        "",
        "| Endpoint | RDS - Random | 95% interval | Verdict |",
        "|---|---:|---:|---|",
    ]
    for endpoint in ("gsm8k", "ood_macro"):
        row = payload["downstream_results"][endpoint]
        lines.append(
            f'| {labels[endpoint]} | {row["difference_percentage_points"]:+.3f} pp '
            f'| [{row["ci95_percentage_points"][0]:+.3f}, '
            f'{row["ci95_percentage_points"][1]:+.3f}] pp | Insufficient evidence |'
        )
    return "\n".join(lines) + "\n"


def expected_outputs() -> dict[str, str]:
    payload = build_payload()
    return {
        "json": render_json(payload),
        "csv": render_csv(payload),
        "markdown": render_markdown(payload),
    }


def write_outputs(outputs: dict[str, str]) -> None:
    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = OUTPUTS[name]
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
        path.write_text(content, encoding="utf-8", newline="\n")


def check_outputs(outputs: dict[str, str]) -> None:
    for name, expected in outputs.items():
        path = OUTPUTS[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = path.read_text(encoding="utf-8")
        if observed != expected:
            raise ValueError(f"public summary differs from audited sources: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = expected_outputs()
    if args.write:
        write_outputs(outputs)
        status = "WRITTEN"
    else:
        check_outputs(outputs)
        status = "PASS"
    print(
        json.dumps(
            {
                "status": status,
                "output_count": len(outputs),
                "source_count": len(SOURCES),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
