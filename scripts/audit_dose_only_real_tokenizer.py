"""Build an immutable CPU-only dose-cap artifact using the formal tokenizer path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from run_b500_formal_resumable import _prepare_training_data  # noqa: E402
from run_budget_equivalent_cell import _resolved_recipe  # noqa: E402

from eg_sft.experiment.identifiable_budget_v4 import (  # noqa: E402
    resolve_identifiable_contract,
)
from eg_sft.experiment.real_tokenizer_dose_audit import (  # noqa: E402
    build_real_tokenizer_dry_run_report,
    canonical_json_sha256,
    write_json_exclusive,
)
from eg_sft.training.b500 import file_sha256  # noqa: E402


DEFAULT_CELL_ID = "dose_rep1_random_free_mix_train17_cap63680"


def _git_text(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _code_bindings() -> dict[str, Any]:
    relative_paths = (
        "scripts/audit_dose_only_real_tokenizer.py",
        "scripts/run_budget_equivalent_cell.py",
        "scripts/run_b500_formal_resumable.py",
        "src/eg_sft/experiment/real_tokenizer_dose_audit.py",
        "src/eg_sft/experiment/identifiable_budget_v4.py",
        "src/eg_sft/training/b500.py",
        "src/eg_sft/training/token_budget.py",
        "src/eg_sft/training/response_only.py",
    )
    file_hashes = {
        relative: file_sha256(ROOT / relative) for relative in relative_paths
    }
    git_head = _git_text("rev-parse", "HEAD")
    git_status = _git_text("status", "--porcelain")
    return {
        "git_head": git_head,
        "git_worktree_dirty": None if git_status is None else bool(git_status),
        "source_package_mode": git_head is None,
        "relevant_file_sha256": file_hashes,
        "relevant_code_bundle_sha256": canonical_json_sha256(file_hashes),
    }


def _tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("formal dry-run requires the fast tokenizer")
    backend_text = tokenizer.backend_tokenizer.to_str()
    return {
        "class": type(tokenizer).__name__,
        "is_fast": True,
        "vocab_size": int(tokenizer.vocab_size),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "backend_tokenizer_sha256": hashlib.sha256(
            backend_text.encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/identifiable_budget_v4_matrix.json"),
    )
    parser.add_argument("--cell-id", default=DEFAULT_CELL_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help=(
            "Optional repository-local tokenizer snapshot for fully offline review. "
            "The frozen repo_id/revision remain recorded as the normative binding."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of accessing the network when the tokenizer/datasets are absent.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = resolve_identifiable_contract(
        repo_root=ROOT,
        config_path=config_path,
        cell_id=str(args.cell_id),
    )
    if contract["study"] != "dose_only":
        raise ValueError("real-tokenizer dry-run only accepts a dose-only cell")
    recipe = _resolved_recipe(
        contract,
        supervision_token_cap=int(contract["supervision_token_cap"]),
        token_cap_policy=str(contract["token_cap_policy"]),
    )
    model_binding = contract["protocol"]["model"]
    if args.tokenizer_path is None:
        tokenizer_source: str | Path = model_binding["repo_id"]
        tokenizer_kwargs = {
            "revision": model_binding["revision"],
            "local_files_only": bool(args.local_files_only),
        }
        tokenizer_load_binding = {
            "source_type": "huggingface_repo",
            "repo_id": model_binding["repo_id"],
            "revision": model_binding["revision"],
            "local_files_only": bool(args.local_files_only),
        }
    else:
        tokenizer_source = args.tokenizer_path.resolve()
        if not tokenizer_source.is_dir():
            raise FileNotFoundError(tokenizer_source)
        try:
            relative_tokenizer_path = tokenizer_source.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(
                "--tokenizer-path must stay inside the repository/review package"
            ) from exc
        tokenizer_kwargs = {"local_files_only": True}
        tokenizer_load_binding = {
            "source_type": "repository_local_snapshot",
            "path": str(relative_tokenizer_path).replace("\\", "/"),
            "normative_repo_id": model_binding["repo_id"],
            "normative_revision": model_binding["revision"],
        }
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples, token_audit, _, _ = _prepare_training_data(
        protocol=contract["protocol"],
        recipe=recipe,
        selected=contract["selection"]["selected"],
        data_manifest_dir=contract["data_dir"],
        tokenizer=tokenizer,
    )

    data_files = {
        filename: {
            "path": str((contract["data_dir"] / filename).relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "sha256": file_sha256(contract["data_dir"] / filename),
        }
        for filename in contract["config"]["data_manifest"]["required_files"]
    }
    bindings = {
        "model": {
            "repo_id": model_binding["repo_id"],
            "revision": model_binding["revision"],
        },
        "tokenizer": _tokenizer_fingerprint(tokenizer),
        "tokenizer_load_binding": tokenizer_load_binding,
        "matrix_config": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": contract["config_sha256"],
        },
        "protocol_config": {
            "path": str(contract["protocol_path"].relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(contract["protocol_path"]),
        },
        "base_recipe_config": {
            "path": str(contract["base_recipe_path"].relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(contract["base_recipe_path"]),
        },
        "selection_manifest": {
            "path": str(contract["selection"]["path"].relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "sha256": contract["selection"]["file_sha256"],
            "selected_id_sha256": contract["selection"]["selected_id_sha256"],
        },
        "data_manifest_files": data_files,
        "code": _code_bindings(),
    }
    report = build_real_tokenizer_dry_run_report(
        cell_id=str(contract["cell_id"]),
        selected=contract["selection"]["selected"],
        tokenized_examples=train_examples,
        token_audit=token_audit,
        epochs=int(recipe["training"]["epochs"]),
        optimizer_steps=int(recipe["training"]["optimizer_steps"]),
        seed=int(contract["seed"]),
        supervision_token_cap=int(contract["supervision_token_cap"]),
        token_cap_policy=str(contract["token_cap_policy"]),
        bindings=bindings,
    )
    write_json_exclusive(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "evidence_type": report["evidence_type"],
                "cell_id": report["cell_id"],
                "output": str(args.output.resolve()),
                "artifact_content_sha256": report["artifact_content_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
