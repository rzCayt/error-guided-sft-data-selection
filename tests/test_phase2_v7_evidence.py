from __future__ import annotations

import pytest

from eg_sft.experiment.phase2_v7_evidence import _safe_member_name


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a\\b"])
def test_evidence_reader_rejects_unsafe_tar_paths(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _safe_member_name(name)


def test_evidence_reader_accepts_ascii_relative_member() -> None:
    _safe_member_name("evaluation/merged/raw_outputs.jsonl")
