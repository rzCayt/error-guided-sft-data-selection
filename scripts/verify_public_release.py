#!/usr/bin/env python3
"""Fail-fast validation for the public research release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".csv", ".py", ".sh", ".toml", ".yml", ".yaml", ".cff", ".html"}
EXCLUDED_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv"}
REQUIRED_PATHS = (
    "README.md",
    "README_CN.md",
    "LICENSE",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "AI_ASSISTANCE.md",
    "CLAIMS_AND_LIMITATIONS.md",
    "REPRODUCE.md",
    "EXPERIMENT_AUDIT.md",
    "EXPERIMENT_AUDIT.json",
    "docs/README.md",
    "docs/research_timeline.md",
    "docs/decision_log.md",
    "docs/code_map.md",
    "docs/current/research_overview_en.md",
    "docs/current/research_overview_zh.md",
    "docs/current/claim_evidence_ledger.md",
    "results/public_summary/main_results.json",
    "results/public_summary/main_results.csv",
    "results/public_summary/main_results_table.md",
    "results/public_summary/experiment_registry.csv",
    "docs/current/results_index.md",
    "docs/current/results_index_zh.md",
    "figures/manifest.json",
    "configs/frozen/MANIFEST.json",
    "releases/v0.5-public-research/README.md",
    "releases/v0.5-public-research/MANIFEST.json",
)
PUBLIC_FACING_ROOT_FILES = {
    "README.md",
    "README_CN.md",
    "THIRD_PARTY_NOTICES.md",
    "AI_ASSISTANCE.md",
    "CLAIMS_AND_LIMITATIONS.md",
    "REPRODUCE.md",
    "EXPERIMENT_AUDIT.md",
    "EXPERIMENT_AUDIT.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_repository_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


def check_required_paths() -> None:
    missing = [text for text in REQUIRED_PATHS if not (ROOT / text).is_file()]
    if missing:
        raise RuntimeError("required public files missing:\n" + "\n".join(missing))


def check_evidence_hashes(summary: dict) -> None:
    for evidence_id, item in summary["evidence"].items():
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing evidence {evidence_id}: {path}")
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"evidence byte-count mismatch: {path}")
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"evidence SHA-256 mismatch: {path}")


def check_readme_results(summary: dict) -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README_CN.md").read_text(encoding="utf-8")
    required_english = (
        "+0.480 percentage points",
        "[-0.954, +1.889]",
        "-0.094 percentage points",
        "[-1.316, +1.149]",
        "24 cells",
        "GPU qualification and formal measurements have not started",
    )
    required_chinese = (
        "+0.480个百分点",
        "[-0.954, +1.889]",
        "-0.094个百分点",
        "[-1.316, +1.149]",
        "24组实验",
        "GPU qualification 和正式测量尚未开始",
    )
    for token in required_english:
        if token not in english:
            raise RuntimeError(f"English README missing canonical token: {token}")
    for token in required_chinese:
        if token not in chinese:
            raise RuntimeError(f"Chinese README missing canonical token: {token}")

    gsm = summary["downstream_results"]["gsm8k"]
    ood = summary["downstream_results"]["ood_macro"]
    if gsm["difference_percentage_points"] != 0.48 or gsm["ci95_percentage_points"] != [-0.954, 1.889]:
        raise RuntimeError("unexpected canonical GSM8K result")
    if ood["difference_percentage_points"] != -0.094 or ood["ci95_percentage_points"] != [-1.316, 1.149]:
        raise RuntimeError("unexpected canonical OOD result")
    if summary["state_dependence_v3"]["gpu_result_available"] is not False:
        raise RuntimeError("public summary incorrectly claims a State Dependence v3 GPU result")


def check_historical_banners() -> None:
    history = ROOT / "docs" / "history"
    files = [path for path in history.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".html"}]
    if not files:
        raise RuntimeError("historical directory contains no documents")
    missing = []
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".md":
            marked = text.startswith("> **Historical snapshot.**") and "历史快照" in text[:300]
        else:
            marked = "<strong>Historical snapshot.</strong>" in text and "历史快照" in text
        if not marked:
            missing.append(path.relative_to(ROOT).as_posix())
    if missing:
        raise RuntimeError("historical documents missing banners:\n" + "\n".join(missing))


def check_sensitive_content(files: list[Path]) -> None:
    secret_patterns = (
        re.compile(r"github" + r"_pat_[A-Za-z0-9_]{12,}"),
        re.compile(r"gh" + r"p_[A-Za-z0-9]{12,}"),
        re.compile(r"hf" + r"_[A-Za-z0-9]{20,}"),
        re.compile(r"sk" + r"-[A-Za-z0-9]{20,}"),
        re.compile(r"ssh\s+-p\s+\d+\s+root@connect\.[^\s]+", re.IGNORECASE),
        re.compile(r"connect\.[A-Za-z0-9.-]*seetacloud\.com", re.IGNORECASE),
    )
    personal_paths = (
        re.compile(r"[A-Za-z]:[\\/]Users[\\/]crz03", re.IGNORECASE),
        re.compile(r"E:[\\/]RA准备", re.IGNORECASE),
    )
    findings: list[str] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        relative = path.relative_to(ROOT).as_posix()
        for pattern in secret_patterns + personal_paths:
            match = pattern.search(text)
            if match:
                findings.append(f"{relative}: {match.group(0)[:80]}")
    if findings:
        raise RuntimeError("sensitive or personal content found:\n" + "\n".join(findings))


def check_public_absolute_paths(files: list[Path]) -> None:
    patterns = (
        re.compile(r"[A-Za-z]:[\\/](?:Users|RA准备)[\\/]", re.IGNORECASE),
        re.compile(r"/(?:root|home)/[A-Za-z0-9_.-]+/"),
    )
    findings: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        is_public = (
            relative in PUBLIC_FACING_ROOT_FILES
            or relative.startswith("docs/current/")
            or relative in {"docs/README.md", "docs/research_timeline.md", "docs/decision_log.md", "docs/code_map.md"}
            or relative.startswith("results/public_summary/")
            or relative.startswith("configs/frozen/")
        )
        if not is_public or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                findings.append(f"{relative}: {match.group(0)}")
    if findings:
        raise RuntimeError("absolute path found in public-facing content:\n" + "\n".join(findings))


def markdown_links(text: str) -> list[str]:
    return re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)


def check_markdown_links() -> None:
    documents = [
        *(ROOT / name for name in PUBLIC_FACING_ROOT_FILES if name.endswith(".md")),
        *(path for path in (ROOT / "docs").glob("*.md")),
        *(path for path in (ROOT / "docs" / "current").glob("*.md")),
    ]
    broken: list[str] = []
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for target_text in markdown_links(text):
            target_text = target_text.strip().strip("<>")
            if target_text.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_text = unquote(target_text.split("#", 1)[0])
            if not target_text:
                continue
            target = (path.parent / target_text).resolve()
            try:
                target.relative_to(ROOT.resolve())
            except ValueError:
                broken.append(f"{path.relative_to(ROOT)} -> outside repo: {target_text}")
                continue
            if not target.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target_text}")
    if broken:
        raise RuntimeError("broken public Markdown links:\n" + "\n".join(broken))


def check_restricted_artifacts(files: list[Path]) -> None:
    forbidden_suffixes = {".safetensors", ".pt", ".pth", ".ckpt"}
    forbidden = [path.relative_to(ROOT).as_posix() for path in files if path.suffix.lower() in forbidden_suffixes]
    if forbidden:
        raise RuntimeError("model/checkpoint files must not be public:\n" + "\n".join(forbidden))
    oversized = [
        f"{path.relative_to(ROOT).as_posix()} ({path.stat().st_size} bytes)"
        for path in files
        if path.stat().st_size > 10 * 1024 * 1024
        and path.relative_to(ROOT).as_posix() != "tests/fixtures/phase2_v7_anchor/training_complete/tokenizer/tokenizer.json"
    ]
    if oversized:
        raise RuntimeError("unexpected files larger than 10 MiB:\n" + "\n".join(oversized))


def check_generated_manifests() -> None:
    figure_manifest = json.loads((ROOT / "figures" / "manifest.json").read_text(encoding="utf-8"))
    source = ROOT / figure_manifest["source"]
    if sha256(source) != figure_manifest["source_sha256"]:
        raise RuntimeError("figure manifest does not match canonical result source")
    for item in figure_manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"figure artifact mismatch: {path}")

    config_manifest = json.loads((ROOT / "configs" / "frozen" / "MANIFEST.json").read_text(encoding="utf-8"))
    for item in config_manifest["files"]:
        source_path = ROOT / item["source"]
        snapshot = ROOT / item["snapshot"]
        expected = item["sha256"]
        if not source_path.is_file() or not snapshot.is_file():
            raise FileNotFoundError(f"frozen-config pair missing: {source_path}, {snapshot}")
        if sha256(source_path) != expected or sha256(snapshot) != expected:
            raise RuntimeError(f"frozen-config pair drift: {item['source']}")


def check_full_release_manifest() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_public_release_manifest.py"), "--check"],
        cwd=ROOT,
        check=True,
    )


def check_internal_audit_disclosure() -> None:
    audit = json.loads((ROOT / "EXPERIMENT_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("independent_external_review") is not False:
        raise RuntimeError("internal audit is incorrectly labeled as independent")
    if audit.get("auditor") != "internal-checklist-fallback":
        raise RuntimeError("unexpected integrity-audit identity")
    if audit.get("overall_verdict") != "warn" or audit.get("integrity_status") != "warn":
        raise RuntimeError("integrity-audit warning boundary changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    check_required_paths()
    check_internal_audit_disclosure()
    summary = json.loads((ROOT / "results" / "public_summary" / "main_results.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != "public-research-summary-v1":
        raise ValueError("unexpected public result schema")
    files = iter_repository_files()
    check_evidence_hashes(summary)
    check_readme_results(summary)
    check_historical_banners()
    check_sensitive_content(files)
    check_public_absolute_paths(files)
    check_markdown_links()
    check_restricted_artifacts(files)
    check_generated_manifests()
    check_full_release_manifest()
    print(f"PASS files={len(files)} evidence={len(summary['evidence'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
