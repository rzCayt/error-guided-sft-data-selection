from __future__ import annotations

import re

NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def parse_numeric(text: str) -> float | None:
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])
