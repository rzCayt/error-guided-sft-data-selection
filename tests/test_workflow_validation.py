from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_workflow_packet.py"


def run_validator(kind: str, path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--kind", kind, "--path", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def load_template(name: str) -> dict:
    path = ROOT / "workflow" / "templates" / name
    return json.loads(path.read_text(encoding="utf-8"))


def write_packet(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_valid_review_package_template_passes() -> None:
    result = run_validator("review_package", ROOT / "workflow" / "templates" / "review_package.json")
    assert result.returncode == 0, result.stderr


def test_valid_review_response_template_passes() -> None:
    result = run_validator("review_response", ROOT / "workflow" / "templates" / "review_response.json")
    assert result.returncode == 0, result.stderr


def test_valid_self_check_template_passes() -> None:
    result = run_validator("self_check", ROOT / "workflow" / "templates" / "main_self_check.json")
    assert result.returncode == 0, result.stderr


def test_review_package_requires_artifacts_and_weaknesses(tmp_path: Path) -> None:
    payload = load_template("review_package.json")
    payload["key_artifacts"] = []
    payload["known_weaknesses"] = []

    result = run_validator("review_package", write_packet(tmp_path, "bad_package.json", payload))

    assert result.returncode != 0
    assert "key_artifacts" in result.stderr


def test_review_package_requires_verification_commands(tmp_path: Path) -> None:
    payload = load_template("review_package.json")
    payload["verification_commands"] = []

    result = run_validator("review_package", write_packet(tmp_path, "bad_package.json", payload))

    assert result.returncode != 0
    assert "verification_commands" in result.stderr


def test_review_package_rejects_failed_verification_command(tmp_path: Path) -> None:
    payload = load_template("review_package.json")
    payload["verification_commands"][0]["status"] = "failed"

    result = run_validator("review_package", write_packet(tmp_path, "bad_package.json", payload))

    assert result.returncode != 0
    assert "failed or not_run" in result.stderr


def test_review_package_must_cover_stage_required_checks(tmp_path: Path) -> None:
    payload = load_template("review_package.json")
    payload["verification_commands"] = payload["verification_commands"][1:]

    result = run_validator("review_package", write_packet(tmp_path, "bad_package.json", payload))

    assert result.returncode != 0
    assert "required checks" in result.stderr


def test_review_package_must_cover_stage_required_artifacts(tmp_path: Path) -> None:
    payload = load_template("review_package.json")
    payload["changed_files"].remove("docs/workflow_cn.md")

    result = run_validator("review_package", write_packet(tmp_path, "bad_package.json", payload))

    assert result.returncode != 0
    assert "required artifacts" in result.stderr


def test_review_package_requires_valid_self_check(tmp_path: Path) -> None:
    payload = load_template("review_package.json")
    payload["main_self_check"] = "missing_self_check.json"

    result = run_validator("review_package", write_packet(tmp_path, "bad_package.json", payload))

    assert result.returncode != 0
    assert "main_self_check" in result.stderr


def test_review_package_requires_matching_stage_ids(tmp_path: Path) -> None:
    stage_plan = load_template("stage_plan.json")
    stage_plan["stage_id"] = "real_base_diagnostic"
    stage_plan_path = write_packet(tmp_path, "stage_plan.json", stage_plan)

    payload = load_template("review_package.json")
    payload["stage_plan"] = str(stage_plan_path)

    result = run_validator("review_package", write_packet(tmp_path, "bad_package.json", payload))

    assert result.returncode != 0
    assert "stage_plan.stage_id" in result.stderr


def test_english_review_response_fails(tmp_path: Path) -> None:
    payload = load_template("review_response.json")
    payload["language"] = "en-US"
    payload["阶段判定"]["结论说明"] = "The workflow can proceed."

    result = run_validator("review_response", write_packet(tmp_path, "english_response.json", payload))

    assert result.returncode != 0
    assert "language must be zh-CN" in result.stderr


def test_review_response_requires_core_evidence_checked(tmp_path: Path) -> None:
    payload = load_template("review_response.json")
    payload["evidence_checked"] = ["changed_files"]

    result = run_validator("review_response", write_packet(tmp_path, "thin_response.json", payload))

    assert result.returncode != 0
    assert "evidence_checked" in result.stderr


def test_review_response_requires_search_records(tmp_path: Path) -> None:
    payload = load_template("review_response.json")
    payload["检索记录"] = []

    result = run_validator("review_response", write_packet(tmp_path, "no_search_response.json", payload))

    assert result.returncode != 0
    assert "检索记录" in result.stderr


def test_review_response_requires_external_sources(tmp_path: Path) -> None:
    payload = load_template("review_response.json")
    payload["外部资料核验"] = []

    result = run_validator("review_response", write_packet(tmp_path, "no_sources_response.json", payload))

    assert result.returncode != 0
    assert "外部资料核验" in result.stderr


def test_review_response_requires_primary_sources_to_allow_next_stage(tmp_path: Path) -> None:
    payload = load_template("review_response.json")
    for source in payload["外部资料核验"]:
        source["primary_source"] = False

    result = run_validator("review_response", write_packet(tmp_path, "weak_sources_response.json", payload))

    assert result.returncode != 0
    assert "2 primary sources" in result.stderr


def test_review_response_requires_source_checked_date_and_summary(tmp_path: Path) -> None:
    payload = load_template("review_response.json")
    del payload["外部资料核验"][0]["checked_date"]

    result = run_validator("review_response", write_packet(tmp_path, "weak_sources_response.json", payload))

    assert result.returncode != 0
    assert "checked_date" in result.stderr


def test_review_package_requires_external_search_queries(tmp_path: Path) -> None:
    payload = load_template("review_package.json")
    payload["external_search_queries"] = []

    result = run_validator("review_package", write_packet(tmp_path, "no_queries_package.json", payload))

    assert result.returncode != 0
    assert "external_search_queries" in result.stderr


def test_review_response_with_blocker_cannot_allow_next_stage(tmp_path: Path) -> None:
    payload = load_template("review_response.json")
    payload["阻塞项"] = ["审查包没有验证命令。"]
    payload["阶段判定"]["allow_next_stage"] = True

    result = run_validator("review_response", write_packet(tmp_path, "blocked_response.json", payload))

    assert result.returncode != 0
    assert "cannot allow next stage" in result.stderr


def test_low_scoring_review_response_cannot_pass(tmp_path: Path) -> None:
    payload = load_template("review_response.json")
    payload["评分表"]["test_leakage_split_contamination"]["score"] = 1
    payload["total_score"] = 14
    payload["阶段判定"]["verdict"] = "pass"
    payload["阶段判定"]["allow_next_stage"] = True

    result = run_validator("review_response", write_packet(tmp_path, "low_score_response.json", payload))

    assert result.returncode != 0
    assert "low scores" in result.stderr or "core minimums" in result.stderr


def test_low_scoring_self_check_cannot_pass(tmp_path: Path) -> None:
    payload = load_template("main_self_check.json")
    payload["scores"]["reproducibility"]["score"] = 1
    payload["total_score"] = 15
    payload["pass"] = True

    result = run_validator("self_check", write_packet(tmp_path, "low_self_check.json", payload))

    assert result.returncode != 0
    assert "self_check cannot pass" in result.stderr
