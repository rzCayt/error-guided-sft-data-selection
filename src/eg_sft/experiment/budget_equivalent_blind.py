"""Blinding controls for the frozen budget-equivalent Phase 1 matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from eg_sft.experiment.budget_equivalent_matrix import validate_matrix_config


BLIND_VERSION = "budget-equivalent-phase1-blind-v1"


def _method_digest(secret: bytes, method: str) -> str:
    return hashlib.sha256(secret + b"\0" + method.encode("utf-8")).hexdigest()


def build_blind_map(
    *, matrix_config: Mapping[str, Any], matrix_sha256: str, secret_hex: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a private mapping and a public manifest with no method names."""

    config = dict(matrix_config)
    validate_matrix_config(config)
    if len(matrix_sha256) != 64:
        raise ValueError("matrix SHA-256 must contain 64 hex characters")
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError as error:
        raise ValueError("blind secret must be hexadecimal") from error
    if len(secret) != 32:
        raise ValueError("blind secret must contain exactly 32 bytes")
    methods = [str(value) for value in config["methods"]]
    ordered = sorted(methods, key=lambda method: _method_digest(secret, method))
    aliases = [f"method_{letter}" for letter in "ABCD"]
    method_to_alias = dict(zip(ordered, aliases, strict=True))
    cells = []
    public_cells = []
    for index, job in enumerate(config["job_order"], start=1):
        blind_cell_id = f"cell_{index:03d}"
        method = str(job["method"])
        private_row = {
            "blind_cell_id": blind_cell_id,
            "cell_id": str(job["cell_id"]),
            "method": method,
            "method_alias": method_to_alias[method],
            "replicate_index": int(job["replicate_index"]),
            "train_seed": int(job["train_seed"]),
        }
        cells.append(private_row)
        public_cells.append(
            {
                "blind_cell_id": blind_cell_id,
                "method_alias": method_to_alias[method],
                "replicate_index": int(job["replicate_index"]),
                "train_seed": int(job["train_seed"]),
            }
        )
    private = {
        "schema_version": BLIND_VERSION,
        "matrix_sha256": matrix_sha256,
        "secret_hex": secret_hex.lower(),
        "method_to_alias": method_to_alias,
        "cells": cells,
        "required_audited_cells_before_unblinding": 16,
    }
    private_bytes = (
        json.dumps(private, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    public = {
        "schema_version": BLIND_VERSION,
        "matrix_sha256": matrix_sha256,
        "private_map_sha256": hashlib.sha256(private_bytes).hexdigest(),
        "method_aliases": aliases,
        "cells": public_cells,
        "actual_method_names_withheld": True,
        "accuracy_withheld_until_all_audits": True,
        "required_audited_cells_before_unblinding": 16,
    }
    return private, public


def blinded_registry(
    *, private_map: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Return registry state without exposing actual methods or cell IDs."""

    by_cell = {str(row["cell_id"]): row for row in registry["jobs"]}
    rows = []
    for mapping in private_map["cells"]:
        source = by_cell[str(mapping["cell_id"])]
        rows.append(
            {
                "blind_cell_id": str(mapping["blind_cell_id"]),
                "method_alias": str(mapping["method_alias"]),
                "replicate_index": int(mapping["replicate_index"]),
                "train_seed": int(mapping["train_seed"]),
                "status": str(source["status"]),
            }
        )
    audited = sum(row["status"] == "AUDITED_PASS" for row in rows)
    required = int(private_map["required_audited_cells_before_unblinding"])
    return {
        "schema_version": "budget-equivalent-phase1-blind-registry-v1",
        "job_count": len(rows),
        "audited_pass_count": audited,
        "unblinding_permitted": audited == required == len(rows),
        "accuracy_withheld": audited != required,
        "jobs": rows,
    }
