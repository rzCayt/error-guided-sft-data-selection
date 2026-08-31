"""Convert an audited READY record into an immutable human-authorized GO record."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from eg_sft.evaluation.phase2_v7_canary import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_json,
    write_exclusive_or_verify,
)
from eg_sft.experiment.phase2_v8_release_gate import HUMAN_CONFIRMATION  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operator_confirmation != HUMAN_CONFIRMATION:
        raise ValueError("exact v8 human release confirmation is required")
    ready_path = args.ready.resolve()
    payload = read_json(ready_path)
    if payload.get("status") != "READY_FOR_HUMAN_REVIEW":
        raise ValueError("v8 release is not ready for human review")
    authorized = {
        **payload,
        "status": "GO",
        "human_authorization": HUMAN_CONFIRMATION,
        "ready_record_sha256": file_sha256(ready_path),
        "approved_at_utc": datetime.now(UTC).isoformat(),
        "formal_matrix_authorized": True,
    }
    write_exclusive_or_verify(args.output.resolve(), canonical_json_bytes(authorized))
    print(json.dumps(authorized, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
