"""CPU-only, immutable adjudication of a batched evaluation qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.evaluation.identifiable_batch_backend import (  # noqa: E402
    compare_backend_rows,
    qualification_decision,
)
from eg_sft.experiment.budget_equivalent_ood_audit_v3 import (  # noqa: E402
    canonical_json_bytes,
    write_bytes_exclusive_or_verify,
)
from eg_sft.training.b500 import file_sha256, read_jsonl  # noqa: E402


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-rows", type=Path, required=True)
    parser.add_argument("--candidate-rows", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read_jsonl(args.reference_rows.resolve())
    candidate = read_jsonl(args.candidate_rows.resolve())
    reference_metrics = _json(args.reference_metrics.resolve())
    candidate_metrics = _json(args.candidate_metrics.resolve())
    comparison = compare_backend_rows(reference=reference, candidate=candidate)
    decision = qualification_decision(
        row_comparison=comparison,
        reference_examples_per_second=float(reference_metrics["examples_per_second"]),
        candidate_examples_per_second=float(candidate_metrics["examples_per_second"]),
        reference_full_cell_seconds=float(reference_metrics["full_cell_seconds"]),
        candidate_full_cell_seconds=float(candidate_metrics["full_cell_seconds"]),
        resume_passed=bool(candidate_metrics.get("resume_passed")),
        non_overwrite_passed=bool(candidate_metrics.get("non_overwrite_passed")),
    )
    artifact = {
        "audit_schema_version": "identifiable-eval-backend-qualification-v1",
        **decision,
        "row_comparison": comparison,
        "inputs": {
            "reference_rows_sha256": file_sha256(args.reference_rows.resolve()),
            "candidate_rows_sha256": file_sha256(args.candidate_rows.resolve()),
            "reference_metrics_sha256": file_sha256(args.reference_metrics.resolve()),
            "candidate_metrics_sha256": file_sha256(args.candidate_metrics.resolve()),
        },
        "accuracy_withheld": True,
        "claim_boundary": "Engineering backend equivalence and throughput only.",
    }
    output = args.output.resolve()
    write_bytes_exclusive_or_verify(output, canonical_json_bytes(artifact))
    write_bytes_exclusive_or_verify(
        output.with_suffix(output.suffix + ".sha256"),
        f"{file_sha256(output)}  {output.name}\n".encode("ascii"),
    )
    print(json.dumps({"status": artifact["status"], "audit_sha256": file_sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
