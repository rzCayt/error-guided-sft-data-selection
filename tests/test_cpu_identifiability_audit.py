from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import pytest

from eg_sft.experiment.cpu_identifiability_audit import (
    failure_taxonomy,
    format_results,
    parser_mismatches,
    run_cpu_identifiability_audit,
    terminal_answer_statement,
    terminal_final_marker,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _raw(record_id: str, text: str, *, gold: str = "7") -> dict:
    from eg_sft.experiment.cpu_identifiability_audit import recompute_frozen_row

    row = {
        "record_id": record_id,
        "gold_value": gold,
        "raw_output": text,
    }
    row.update(recompute_frozen_row(row))
    return row


def _cell(root: Path, *, cell_id: str = "rep1_random_free_mix_train17") -> Path:
    cell = root / cell_id
    manifest = {
        "seed": 17,
        "config": {
            "cell_id": cell_id,
            "method": "random_free_mix",
            "replicate_index": 1,
            "selection_manifest_sha256": "selection-sha",
            "selected_id_sha256": "selected-sha",
        },
    }
    _write_json(cell / "manifest.json", manifest)
    _write_json(cell / "audit/formal_cell_audit.json", {"status": "PASS"})
    _write_json(cell / "audit/ood_audit.json", {"status": "PASS"})
    token_rows = [
        {
            "candidate_id": "c1",
            "supervised_tokens": 7,
            "total_tokens": 20,
        },
        {
            "candidate_id": "c2",
            "supervised_tokens": 9,
            "total_tokens": 30,
        },
    ]
    _write_json(cell / "training_complete/token_audit.json", token_rows)
    _write_json(
        cell / "training_complete/token_budget_audit.json",
        {"response_supervision_exposure_tokens": 32, "optimizer_steps": 64},
    )
    rows = [
        _raw("a", "work\nFinal answer: 7"),
        _raw("b", "work. Final answer: 7"),
        _raw("c", "work\nAnswer: 7"),
        _raw("d", "work\nFinal answer: 7 apples"),
    ]
    for dataset, member in {
        "gsm8k": "evaluation/merged/raw_outputs.jsonl",
        "svamp": "evaluation/ood/svamp/merged/raw_outputs.jsonl",
        "asdiv_numeric": "evaluation/ood/asdiv_numeric/merged/raw_outputs.jsonl",
        "multiarith": "evaluation/ood/multiarith/merged/raw_outputs.jsonl",
    }.items():
        _write_jsonl(cell / member, [{**row, "dataset": dataset} for row in rows])
    return cell


def test_format_criteria_are_additive_and_do_not_relax_frozen_parser() -> None:
    text = "reasoning. Final answer: 7"
    results = format_results(text)
    assert not results["frozen_strict_standalone_line"].ok
    assert results["terminal_final_marker_suffix"].ok
    assert results["terminal_explicit_answer_statement"].ok
    assert failure_taxonomy(text, results) == "embedded_or_decorated_marker_on_final_line"
    assert terminal_final_marker("Final answer: 7 apples").status == "marker_or_payload_not_terminal"
    assert terminal_answer_statement("**Answer:** **7**").ok


def test_taxonomy_distinguishes_missing_invalid_extra_and_multiple() -> None:
    assert failure_taxonomy("answer is 7") == "missing_final_marker_with_numeric"
    assert failure_taxonomy("Final answer: <number>") == "invalid_final_payload"
    assert failure_taxonomy("Final answer: 7 apples") == "extra_text_after_final_numeric"
    assert failure_taxonomy("Final answer: 7\nFinal answer: 8") == "multiple_final_markers"


def test_parser_recomputation_detects_changed_stored_metric() -> None:
    row = _raw("a", "Final answer: 7")
    assert parser_mismatches(row) == []
    row["numeric_correct"] = False
    assert parser_mismatches(row) == ["numeric_correct"]


def test_directory_and_tar_inputs_produce_non_overwriting_artifact(tmp_path: Path) -> None:
    cell = _cell(tmp_path / "cells")
    output = run_cpu_identifiability_audit(
        inputs=[cell],
        output_root=tmp_path / "out",
        expected_cells=1,
        expected_dataset_counts=False,
        sample_size=5,
        run_id="directory-run",
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["parser_mismatch_count"] == 0
    assert summary["raw_output_count"] == 16
    assert summary["failure_sample_count"] == 5
    assert len(summary["analysis_code"]["frozen_parser_sha256"]) == 64
    assert len(summary["inputs"][0]["sha256"]) == 64
    assert (output / "format_criteria.csv").is_file()
    assert (output / "report_cn.md").is_file()
    assert (output / "estimand_note.md").is_file()
    generation_rows = [
        json.loads(line)
        for line in (output / "generation_lengths.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(generation_rows) == 16
    assert generation_rows[0]["primary_length_measure"] == "generation_char_length_proxy"
    assert "generation_char_length" in generation_rows[0]
    assert "generation_line_count" in generation_rows[0]
    composition_text = (output / "selection_composition.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "prompt_token_band" in composition_text
    assert "prompt_tokens" in composition_text.splitlines()[0]
    public_text = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in output.iterdir()
        if path.is_file()
    )
    assert str(cell.resolve()) not in public_text
    assert re.search(r"[A-Za-z]:[\\/]", public_text) is None
    with pytest.raises(FileExistsError):
        run_cpu_identifiability_audit(
            inputs=[cell],
            output_root=tmp_path / "out",
            expected_cells=1,
            expected_dataset_counts=False,
            run_id="directory-run",
        )

    archive = tmp_path / "cell.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in cell.rglob("*"):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(cell).as_posix())
    tar_output = run_cpu_identifiability_audit(
        inputs=[archive],
        output_root=tmp_path / "out",
        expected_cells=1,
        expected_dataset_counts=False,
        sample_size=2,
        run_id="tar-run",
    )
    assert (tar_output / "artifact_manifest.json").is_file()


def test_audit_fails_closed_on_parser_drift(tmp_path: Path) -> None:
    cell = _cell(tmp_path / "cells")
    raw_path = cell / "evaluation/merged/raw_outputs.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["strict_parse_status"] = "missing_final_marker"
    _write_jsonl(raw_path, rows)
    with pytest.raises(ValueError, match="frozen parser recomputation mismatch"):
        run_cpu_identifiability_audit(
            inputs=[cell],
            output_root=tmp_path / "out",
            expected_cells=1,
            expected_dataset_counts=False,
            run_id="must-fail",
        )
