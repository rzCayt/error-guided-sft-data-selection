#!/usr/bin/env python3
"""Add, migrate, or verify explicit banners on historical documents."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "docs" / "history"

MARKDOWN_BANNER = (
    "> **Historical snapshot.**\n>\n"
    "> This document records an earlier research stage and is not the current result.\n>\n"
    "> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**\n\n"
)
LEGACY_MARKDOWN_BANNER = (
    "> **Historical snapshot.**  \n"
    "> This document records an earlier research stage and is not the current result.  \n"
    "> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**\n\n"
)
HTML_MARKER = "<strong>Historical snapshot.</strong> This document records an earlier research stage"
HTML_BANNER = (
    '<div role="note" style="border:2px solid #b45309;padding:12px;margin:12px 0;">'
    "<strong>Historical snapshot.</strong> This document records an earlier research stage "
    "and is not the current result.<br><strong>历史快照：本文档记录早期研究阶段，"
    "不代表当前研究结论。</strong></div>\n"
)


def historical_files() -> list[Path]:
    if not HISTORY.is_dir():
        raise FileNotFoundError(f"missing history directory: {HISTORY}")
    return sorted(
        path
        for path in HISTORY.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".html"}
    )


def is_marked(path: Path, text: str) -> bool:
    if path.suffix.lower() == ".md":
        return text.startswith(MARKDOWN_BANNER)
    return HTML_MARKER in text


def add_or_migrate_banner(path: Path, text: str) -> str:
    if path.suffix.lower() == ".md":
        if text.startswith(LEGACY_MARKDOWN_BANNER):
            return MARKDOWN_BANNER + text[len(LEGACY_MARKDOWN_BANNER) :]
        return MARKDOWN_BANNER + text.lstrip("\ufeff")
    lower = text.lower()
    body_position = lower.find("<body")
    if body_position < 0:
        return HTML_BANNER + text
    close = text.find(">", body_position)
    if close < 0:
        raise ValueError(f"malformed HTML body tag: {path}")
    return text[: close + 1] + "\n" + HTML_BANNER + text[close + 1 :]


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    files = historical_files()
    if not files:
        raise RuntimeError("no historical Markdown/HTML files found")

    missing: list[str] = []
    changed = 0
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if is_marked(path, text):
            continue
        if args.check:
            missing.append(path.relative_to(ROOT).as_posix())
            continue
        path.write_text(add_or_migrate_banner(path, text), encoding="utf-8", newline="\n")
        changed += 1

    if missing:
        raise RuntimeError("historical banner missing:\n" + "\n".join(missing))
    print(f"PASS files={len(files)} changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
