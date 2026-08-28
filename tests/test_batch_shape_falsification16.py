from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from eg_sft.evaluation.batch_shape_falsification16 import (
    audit_phase,
    canonical_token_ids,
    compare_rows,
    derive_selection,
    effective_token_count,
    read_json,
    validate_config,
    validate_selection_against_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "batch_shape_falsification16_v1.json"
EOS = 151643


def _load_runner_module():
    import importlib.util

    path = ROOT / "scripts" / "run_batch_shape_falsification16.py"
    spec = importlib.util.spec_from_file_location("batch_shape_falsification16_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _row(record_id: str, value: str = "1", token: int = 10) -> dict:
    return {
        "record_id": record_id,
        "source_index": int(record_id.rsplit("-", 1)[-1], 16) % 10000,
        "generated_token_ids": [token, EOS],
        "raw_output": f"Final answer: {value}",
        "parsed_prediction": value,
        "numeric_correct": value == "1",
        "strict_parse_status": "ok",
        "parse_mode": "strict_final_marker",
        "parse_status": "ok",
    }


def _synthetic_sources(config: dict) -> dict[int, list[dict]]:
    persistent = list(config["selection"]["persistent_semantic_mismatch_ids"])
    controls = list(config["selection"]["lowest_risk_control_ids"])
    filler = [f"gsm8k-test-{index + 2000:04d}-{index:012x}" for index in range(112)]
    ids = [*persistent, *controls, *filler]
    reference = [_row(record_id) for record_id in ids]
    rows = {1: copy.deepcopy(reference)}
    for batch in (2, 4, 8):
        candidate = copy.deepcopy(reference)
        for index in range(12):
            candidate[index]["generated_token_ids"] = [100 + batch + index, EOS]
            candidate[index]["raw_output"] = f"Final answer: {batch + index}"
            candidate[index]["parsed_prediction"] = str(batch + index)
            candidate[index]["numeric_correct"] = False
        # Synthetic filler must rank below the four frozen controls.
        for index in range(16, len(candidate)):
            if batch == 2:
                candidate[index]["generated_token_ids"] = [500 + index, EOS]
                candidate[index]["raw_output"] = f"Final answer: {500 + index}"
                candidate[index]["parsed_prediction"] = str(500 + index)
                candidate[index]["numeric_correct"] = False
        rows[batch] = candidate
    # Three controls are fully equal. The fourth has trailing EOS only, matching
    # the frozen lowest-risk ordering.
    for batch in (2, 4, 8):
        rows[batch][15]["generated_token_ids"] = [10, EOS, EOS]
    return rows


def _pass_row(row: dict) -> dict:
    copied = copy.deepcopy(row)
    copied["raw_generated_tensor_ids"] = list(copied["generated_token_ids"])
    copied["canonical_generated_ids"] = canonical_token_ids(
        copied["generated_token_ids"], EOS
    )
    return copied


def _write_pass(root: Path, pass_id: str, rows: list[dict]) -> None:
    path = root / "runs" / pass_id / "raw_outputs.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(_pass_row(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_canonical_ids_remove_trailing_eos_padding() -> None:
    assert canonical_token_ids([5, EOS, EOS, EOS], EOS) == [5, EOS]
    assert effective_token_count([5, EOS, EOS], EOS) == 1


def test_compare_rows_separates_raw_padding_from_l0_to_l4() -> None:
    left = _row("gsm8k-test-0000-000000000000")
    right = copy.deepcopy(left)
    right["generated_token_ids"] = [10, EOS, EOS, EOS]
    report = compare_rows(reference=[left], candidate=[right], eos_token_id=EOS)
    assert report["exact_l0_to_l4"] is True
    assert report["l0_canonical_token_mismatch"] == 0


def test_frozen_config_and_selection_are_rederived() -> None:
    config = _config()
    validate_config(config)
    selection = derive_selection(
        source_rows=_synthetic_sources(config),
        eos_token_id=EOS,
    )
    validate_selection_against_config(selection=selection, config=config)
    assert selection["selected_count"] == 16
    assert len(selection["strata"]["persistent_semantic_mismatch"]) == 12
    assert len(selection["strata"]["lowest_risk_control"]) == 4


def test_phase_audits_never_authorize_batch_gt_one(tmp_path: Path) -> None:
    config = _config()
    sources = _synthetic_sources(config)
    selection = derive_selection(source_rows=sources, eos_token_id=EOS)
    by_id = {row["record_id"]: row for row in sources[1]}
    selected = [
        copy.deepcopy(by_id[record_id])
        for record_id in selection["selected_record_ids"]
    ]
    for pass_id in (
        "bf16_b1_natural_repeat",
        "bf16_b4_fixed_a",
        "bf16_b4_fixed_b",
        "bf16_b1_fixed",
        "fp32_b1_fixed",
        "fp32_b4_fixed",
    ):
        _write_pass(tmp_path, pass_id, selected)

    for phase in ("baseline_repeat", "bf16_repeat", "width_effect"):
        report = audit_phase(
            phase=phase,
            output_root=tmp_path,
            source_rows=sources,
            selection=selection,
            config=config,
        )
        assert report["decision"] == "CONTINUE"
        assert report["batch_gt_1_authorized"] is False

    final = audit_phase(
        phase="final_mechanism",
        output_root=tmp_path,
        source_rows=sources,
        selection=selection,
        config=config,
    )
    assert final["decision"] == "LIMITED_REQUALIFICATION_CANDIDATE"
    assert final["batch_gt_1_authorized"] is False
    assert final["accuracy_aggregated"] is False


def test_fp32_three_mismatches_only_allow_limited_debug(tmp_path: Path) -> None:
    config = _config()
    sources = _synthetic_sources(config)
    selection = derive_selection(source_rows=sources, eos_token_id=EOS)
    by_id = {row["record_id"]: row for row in sources[1]}
    selected = [
        copy.deepcopy(by_id[record_id])
        for record_id in selection["selected_record_ids"]
    ]
    for pass_id in ("bf16_b1_fixed", "fp32_b1_fixed"):
        _write_pass(tmp_path, pass_id, selected)
    bf16_batch = copy.deepcopy(selected)
    for index in range(3):
        bf16_batch[index]["generated_token_ids"] = [700 + index, EOS]
        bf16_batch[index]["raw_output"] = f"Final answer: {700 + index}"
        bf16_batch[index]["parsed_prediction"] = str(700 + index)
        bf16_batch[index]["numeric_correct"] = False
    _write_pass(tmp_path, "bf16_b4_fixed_a", bf16_batch)
    fp32_batch = copy.deepcopy(selected)
    for index in range(3):
        fp32_batch[index]["generated_token_ids"] = [900 + index, EOS]
        fp32_batch[index]["raw_output"] = f"Final answer: {900 + index}"
        fp32_batch[index]["parsed_prediction"] = str(900 + index)
        fp32_batch[index]["numeric_correct"] = False
    _write_pass(tmp_path, "fp32_b4_fixed", fp32_batch)
    report = audit_phase(
        phase="final_mechanism",
        output_root=tmp_path,
        source_rows=sources,
        selection=selection,
        config=config,
    )
    assert report["decision"] == "LIMITED_DEBUG_ALLOWED"
    assert report["batch_gt_1_authorized"] is False


def test_read_json_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text("[]", encoding="utf-8")
    try:
        read_json(path)
    except ValueError as error:
        assert "JSON object" in str(error)
    else:
        raise AssertionError("expected object validation failure")


def test_offline_snapshot_must_match_frozen_revision(tmp_path: Path) -> None:
    runner = _load_runner_module()
    config = _config()
    snapshot = tmp_path / config["model"]["revision"]
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with patch.dict(
        os.environ,
        {"EG_SFT_OFFLINE_MODEL_SNAPSHOT": str(snapshot)},
        clear=False,
    ):
        source, kwargs, manifest = runner._frozen_pretrained_source(
            config=config, section="model"
        )
    assert source == str(snapshot.resolve())
    assert kwargs == {"local_files_only": True}
    assert manifest["source_type"] == "frozen_local_snapshot"
    assert "required_file_sha256" in manifest
    assert str(snapshot.resolve()) not in json.dumps(manifest)


def test_offline_snapshot_rejects_wrong_revision(tmp_path: Path) -> None:
    runner = _load_runner_module()
    config = _config()
    snapshot = tmp_path / "wrong-revision"
    snapshot.mkdir()
    with patch.dict(
        os.environ,
        {"EG_SFT_OFFLINE_MODEL_SNAPSHOT": str(snapshot)},
        clear=False,
    ):
        with pytest.raises(ValueError, match="frozen revision"):
            runner._frozen_pretrained_source(config=config, section="tokenizer")
