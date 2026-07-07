from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGES_PATH = ROOT / "workflow" / "stages.json"

VALID_KINDS = {"stage_plan", "review_package", "review_response", "self_check"}
VALID_VERDICTS = {"pass", "conditional_pass", "fail"}


class ValidationError(Exception):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("top-level JSON value must be an object")
    return data


def load_stage_config() -> dict[str, Any]:
    return load_json(STAGES_PATH)


def require_fields(data: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValidationError(f"missing required fields: {', '.join(missing)}")


def require_nonempty_list(data: dict[str, Any], field: str) -> list[Any]:
    value = data.get(field)
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{field} must be a non-empty list")
    return value


def require_language_zh(data: dict[str, Any]) -> None:
    if data.get("language") not in {"zh-CN", "中文"}:
        raise ValidationError("language must be zh-CN")
    if not contains_cjk(data):
        raise ValidationError("packet must contain Chinese text")


def contains_cjk(value: Any) -> bool:
    if isinstance(value, str):
        return any("\u4e00" <= char <= "\u9fff" for char in value)
    if isinstance(value, list):
        return any(contains_cjk(item) for item in value)
    if isinstance(value, dict):
        return any(contains_cjk(key) or contains_cjk(item) for key, item in value.items())
    return False


def known_stage_ids(config: dict[str, Any]) -> set[str]:
    stages = config.get("stages")
    if not isinstance(stages, list):
        raise ValidationError("workflow/stages.json must contain stages list")
    return {stage["id"] for stage in stages if isinstance(stage, dict) and "id" in stage}


def stage_by_id(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in config.get("stages", []):
        if isinstance(stage, dict) and stage.get("id") == stage_id:
            return stage
    raise ValidationError(f"unknown stage_id: {stage_id}")


def validate_stage_id(data: dict[str, Any], config: dict[str, Any]) -> None:
    stage_id = data.get("stage_id")
    if stage_id not in known_stage_ids(config):
        raise ValidationError(f"unknown stage_id: {stage_id}")


def validate_reviewable_stage(data: dict[str, Any], config: dict[str, Any]) -> None:
    validate_stage_id(data, config)
    reviewable = set(config.get("reviewable_stage_ids", []))
    current = config.get("current_allowed_stage")
    if current:
        reviewable.add(current)
    if data.get("stage_id") not in reviewable:
        raise ValidationError(f"stage_id is not reviewable now: {data.get('stage_id')}")


def validate_paths_exist(paths: list[str], field: str) -> None:
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        raise ValidationError(f"{field} references missing paths: {', '.join(missing)}")


def validate_stage_plan(data: dict[str, Any], config: dict[str, Any]) -> None:
    require_fields(
        data,
        [
            "schema_version",
            "kind",
            "language",
            "stage_id",
            "stage_goal",
            "planned_changes",
            "required_artifacts",
            "required_checks",
            "forbidden_claims",
            "stop_conditions",
            "next_stage_if_passed",
        ],
    )
    require_language_zh(data)
    validate_stage_id(data, config)
    require_nonempty_list(data, "planned_changes")
    require_nonempty_list(data, "required_artifacts")
    require_nonempty_list(data, "required_checks")
    require_nonempty_list(data, "forbidden_claims")
    require_nonempty_list(data, "stop_conditions")


def validate_review_package(data: dict[str, Any], config: dict[str, Any]) -> None:
    require_fields(
        data,
        [
            "schema_version",
            "kind",
            "language",
            "stage_id",
            "stage_goal",
            "stage_plan",
            "main_self_check",
            "changed_files",
            "key_artifacts",
            "verification_commands",
            "key_results",
            "known_weaknesses",
            "questions_for_reviewer",
            "required_output_language",
            "required_reviewer_steps",
            "requested_verdicts",
        ],
    )
    require_language_zh(data)
    validate_reviewable_stage(data, config)
    stage_config = stage_by_id(config, data["stage_id"])
    stage_plan_path = ROOT / str(data["stage_plan"])
    self_check_path = ROOT / str(data["main_self_check"])
    if not stage_plan_path.exists():
        raise ValidationError(f"stage_plan path does not exist: {data['stage_plan']}")
    if not self_check_path.exists():
        raise ValidationError(f"main_self_check path does not exist: {data['main_self_check']}")
    stage_plan = load_json(stage_plan_path)
    self_check = load_json(self_check_path)
    validate_stage_plan(stage_plan, config)
    validate_self_check(self_check, config)
    if stage_plan.get("stage_id") != data["stage_id"]:
        raise ValidationError("stage_plan.stage_id must match review_package.stage_id")
    if self_check.get("stage_id") != data["stage_id"]:
        raise ValidationError("main_self_check.stage_id must match review_package.stage_id")

    changed_files = require_nonempty_list(data, "changed_files")
    changed_file_paths = [str(path) for path in changed_files]
    validate_paths_exist(changed_file_paths, "changed_files")

    artifacts = require_nonempty_list(data, "key_artifacts")
    artifact_paths = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not artifact.get("path"):
            raise ValidationError("each key_artifacts item must include path")
        artifact_paths.append(str(artifact["path"]))
    validate_paths_exist(artifact_paths, "key_artifacts")
    covered_artifacts = set(changed_file_paths) | set(artifact_paths)
    missing_required_artifacts = [
        path for path in stage_config.get("required_artifacts", []) if path not in covered_artifacts
    ]
    if missing_required_artifacts:
        raise ValidationError(
            "review_package does not cover required artifacts: "
            + ", ".join(missing_required_artifacts)
        )

    commands = require_nonempty_list(data, "verification_commands")
    command_texts = []
    for command in commands:
        if not isinstance(command, dict):
            raise ValidationError("each verification_commands item must be an object")
        if not command.get("command") or command.get("status") not in {"passed", "failed", "not_run"}:
            raise ValidationError("each verification command needs command and valid status")
        if command.get("status") != "passed":
            raise ValidationError("review_package cannot pass validation with failed or not_run commands")
        if not command.get("output_summary"):
            raise ValidationError("each verification command needs output_summary")
        command_texts.append(command["command"])
    missing_required_checks = [
        command for command in stage_config.get("required_checks", []) if command not in command_texts
    ]
    if missing_required_checks:
        raise ValidationError(
            "review_package does not cover required checks: " + ", ".join(missing_required_checks)
        )

    require_nonempty_list(data, "key_results")
    require_nonempty_list(data, "known_weaknesses")
    require_nonempty_list(data, "questions_for_reviewer")
    require_nonempty_list(data, "required_reviewer_steps")
    require_nonempty_list(data, "requested_verdicts")
    if data.get("required_output_language") != "zh-CN":
        raise ValidationError("review package must require zh-CN reviewer output")


def score_total(scores: dict[str, Any], criteria: dict[str, int]) -> int:
    total = 0
    for name, max_score in criteria.items():
        item = scores.get(name)
        if not isinstance(item, dict):
            raise ValidationError(f"missing score item: {name}")
        score = item.get("score")
        declared_max = item.get("max_score")
        if not isinstance(score, int) or not isinstance(declared_max, int):
            raise ValidationError(f"{name} score and max_score must be integers")
        if declared_max != max_score:
            raise ValidationError(f"{name} max_score must be {max_score}")
        if score < 0 or score > max_score:
            raise ValidationError(f"{name} score out of range")
        if not item.get("reason") or not contains_cjk(item["reason"]):
            raise ValidationError(f"{name} must include Chinese reason")
        total += score
    return total


def enforce_core_minimums(scores: dict[str, Any], core_minimums: dict[str, int]) -> bool:
    return all(scores[name]["score"] >= minimum for name, minimum in core_minimums.items())


def validate_self_check(data: dict[str, Any], config: dict[str, Any]) -> None:
    require_fields(
        data,
        [
            "schema_version",
            "kind",
            "language",
            "stage_id",
            "hard_blockers",
            "scores",
            "total_score",
            "pass",
            "next_action",
        ],
    )
    require_language_zh(data)
    validate_stage_id(data, config)
    if not isinstance(data["hard_blockers"], list):
        raise ValidationError("hard_blockers must be a list")
    rubric = config["main_self_check_rubric"]
    total = score_total(data["scores"], rubric["criteria"])
    if data["total_score"] != total:
        raise ValidationError(f"total_score must be {total}")
    core_ok = enforce_core_minimums(data["scores"], rubric["core_minimums"])
    pass_allowed = not data["hard_blockers"] and total >= rubric["threshold_total"] and core_ok
    if data["pass"] and not pass_allowed:
        raise ValidationError("self_check cannot pass with blockers, low total, or low core scores")


def validate_review_response(data: dict[str, Any], config: dict[str, Any]) -> None:
    require_fields(
        data,
        [
            "schema_version",
            "kind",
            "language",
            "reviewer_role",
            "evidence_checked",
            "阻塞项",
            "主要问题",
            "次要问题",
            "必修复",
            "评分表",
            "total_score",
            "阶段判定",
        ],
    )
    require_language_zh(data)
    for field in ["阻塞项", "主要问题", "次要问题", "必修复"]:
        if not isinstance(data[field], list):
            raise ValidationError(f"{field} must be a list")
    evidence_checked = set(str(item) for item in require_nonempty_list(data, "evidence_checked"))
    required_evidence = {"changed_files", "key_artifacts", "verification_commands"}
    if not required_evidence.issubset(evidence_checked):
        missing = ", ".join(sorted(required_evidence - evidence_checked))
        raise ValidationError(f"evidence_checked missing required evidence: {missing}")

    rubric = config["reviewer_rubric"]
    total = score_total(data["评分表"], rubric["criteria"])
    if data["total_score"] != total:
        raise ValidationError(f"total_score must be {total}")
    core_ok = enforce_core_minimums(data["评分表"], rubric["core_minimums"])

    decision = data["阶段判定"]
    if not isinstance(decision, dict):
        raise ValidationError("阶段判定 must be an object")
    require_fields(decision, ["verdict", "allow_next_stage", "allowed_next_stage", "结论说明"])
    if decision["verdict"] not in VALID_VERDICTS:
        raise ValidationError(f"invalid verdict: {decision['verdict']}")
    if not isinstance(decision["allow_next_stage"], bool):
        raise ValidationError("allow_next_stage must be boolean")
    if not contains_cjk(decision["结论说明"]):
        raise ValidationError("阶段判定.结论说明 must be Chinese")

    has_blockers = bool(data["阻塞项"])
    pass_allowed = not has_blockers and total >= rubric["threshold_total"] and core_ok
    if decision["allow_next_stage"] and not pass_allowed:
        raise ValidationError("review_response cannot allow next stage with blockers or low scores")
    if has_blockers and decision["allow_next_stage"]:
        raise ValidationError("review_response cannot allow next stage when blockers exist")
    if decision["verdict"] == "pass" and not pass_allowed:
        raise ValidationError("pass verdict requires no blockers, threshold total, and core minimums")


def validate(kind: str, path: Path) -> None:
    if kind not in VALID_KINDS:
        raise ValidationError(f"kind must be one of: {', '.join(sorted(VALID_KINDS))}")
    data = load_json(path)
    if data.get("kind") != kind and not (kind == "self_check" and data.get("kind") == "main_self_check"):
        raise ValidationError(f"packet kind mismatch: expected {kind}, got {data.get('kind')}")
    config = load_stage_config()
    if kind == "stage_plan":
        validate_stage_plan(data, config)
    elif kind == "review_package":
        validate_review_package(data, config)
    elif kind == "review_response":
        validate_review_response(data, config)
    elif kind == "self_check":
        validate_self_check(data, config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=sorted(VALID_KINDS))
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    path = (ROOT / args.path).resolve()
    try:
        validate(args.kind, path)
    except ValidationError as exc:
        print(f"workflow packet invalid: {exc}", file=sys.stderr)
        return 1
    print(f"workflow packet valid: {args.kind} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
