"""CPU-only diagnostics for the audited Phase 1A evidence.

This module is deliberately additive.  It imports the frozen GSM8K parser and
recomputes its outputs, but it never changes the parser or the primary metrics.
The additional format criteria are descriptive sensitivity checks only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Iterator

import eg_sft.gsm8k.parser as frozen_parser_module
from eg_sft.gsm8k.parser import parse_generated_answer, parse_last_numeric_answer


SCHEMA_VERSION = "phase1a-cpu-identifiability-audit-v1"
DATASET_MEMBERS = {
    "gsm8k": "evaluation/merged/raw_outputs.jsonl",
    "svamp": "evaluation/ood/svamp/merged/raw_outputs.jsonl",
    "asdiv_numeric": "evaluation/ood/asdiv_numeric/merged/raw_outputs.jsonl",
    "multiarith": "evaluation/ood/multiarith/merged/raw_outputs.jsonl",
}
EXPECTED_DATASET_COUNTS = {
    "gsm8k": 1319,
    "svamp": 300,
    "asdiv_numeric": 2067,
    "multiarith": 155,
}
ESTIMAND_NOTE = """# Estimand 与结论边界

- RDS 在当前实验中定义为一个固定的条件选择政策，而不是从同一随机化机制中反复抽样的方法。
- 多份 random 名单用于估计 random policy expectation 与其名单方差。
- RDS query-bootstrap replicate 只能解释为查询重抽样敏感性，不当作四个完全独立的政策实现。
- common/free 对比是组成约束下与自由组成下的描述性对比，不构成严格的因果中介分解。
- 本审计的放宽格式判据、生成长度和四象限仅用于诊断，不替代冻结 parser 的主指标。
"""

_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
_ANY_NUMBER = re.compile(_NUMBER)
_FINAL_MARKER = re.compile(r"Final\s+answer\s*:", re.IGNORECASE)
_PREFIX_NUMBER = re.compile(rf"^\s*(?P<number>{_NUMBER})")
_TERMINAL_FINAL_MARKER = re.compile(
    rf"Final\s+answer\s*:\s*(?P<number>{_NUMBER})\s*[.!]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_ANSWER_FAMILY = re.compile(
    rf"(?:"
    rf"(?:Final\s+answer|Answer)\s*:\s*(?P<number_colon>{_NUMBER})"
    rf"|(?:Therefore|Thus|Hence)\s*,?\s*(?:the\s+)?answer\s+is\s+"
    rf"(?P<number_statement>{_NUMBER})"
    rf")\s*[.!]?\s*$",
    re.IGNORECASE,
)
_ONLY_DECORATION = re.compile(r"^[\s.!;,`*_\[\](){}]*$")


@dataclass(frozen=True)
class FormatResult:
    """Result of one additional, non-primary format criterion."""

    ok: bool
    status: str
    value: Decimal | None = None


@dataclass
class CellData:
    cell_id: str
    method: str
    replicate_index: int
    train_seed: int
    manifest: dict[str, Any]
    token_audit: list[dict[str, Any]]
    token_budget_audit: dict[str, Any]
    outputs: dict[str, list[dict[str, Any]]]
    source_path: Path
    source_sha256: str | None
    selection_manifest: dict[str, Any] | None = None


class _Reader:
    def exists(self, member: str) -> bool:
        raise NotImplementedError

    def read_bytes(self, member: str) -> bytes:
        raise NotImplementedError


class _DirectoryReader(_Reader):
    def __init__(self, root: Path) -> None:
        self.root = root

    def exists(self, member: str) -> bool:
        return (self.root / member).is_file()

    def read_bytes(self, member: str) -> bytes:
        return (self.root / member).read_bytes()


class _TarReader(_Reader):
    def __init__(self, archive: tarfile.TarFile) -> None:
        self.archive = archive
        members = archive.getmembers()
        names = [member.name for member in members if member.isfile()]
        if len(names) != len(set(names)):
            raise ValueError("evidence archive contains duplicate member names")
        self.names = set(names)

    def exists(self, member: str) -> bool:
        return member in self.names

    def read_bytes(self, member: str) -> bytes:
        handle = self.archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(member)
        return handle.read()


@contextmanager
def _open_reader(source: Path) -> Iterator[_Reader]:
    if source.is_dir():
        yield _DirectoryReader(source)
        return
    with tarfile.open(source, mode="r:gz") as archive:
        yield _TarReader(archive)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_artifact_label(path: Path) -> str:
    """Return a stable provenance label without exposing a host filesystem path."""

    suffix = "extracted-cell" if path.is_dir() else "evidence-archive"
    return f"{suffix}:{path.name}"


def _directory_input_sha256(root: Path, members: Iterable[str]) -> str:
    """Hash the exact directory members consumed by this audit."""

    digest = hashlib.sha256()
    for member in sorted(members):
        digest.update(member.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(root / member)))
    return digest.hexdigest()


def _json(reader: _Reader, member: str) -> Any:
    return json.loads(reader.read_bytes(member).decode("utf-8"))


def _jsonl(reader: _Reader, member: str) -> list[dict[str, Any]]:
    text = reader.read_bytes(member).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value).replace(",", ""))
    except InvalidOperation:
        return None


def _same_decimal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _decimal(left) == _decimal(right)


def terminal_final_marker(text: str) -> FormatResult:
    """Accept one ``Final answer: number`` suffix, even after same-line prose."""

    if not text.strip():
        return FormatResult(False, "empty_output")
    markers = list(_FINAL_MARKER.finditer(text))
    if not markers:
        return FormatResult(False, "missing_final_marker")
    if len(markers) != 1:
        return FormatResult(False, "multiple_final_markers")
    match = _TERMINAL_FINAL_MARKER.search(text)
    if match is None:
        return FormatResult(False, "marker_or_payload_not_terminal")
    return FormatResult(True, "ok", _decimal(match.group("number")))


def terminal_answer_statement(text: str) -> FormatResult:
    """Accept a small family of explicit terminal answer statements.

    Markdown emphasis is removed only for this sensitivity criterion.  The
    frozen strict parser and primary metrics remain untouched.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return FormatResult(False, "empty_output")
    normalized = re.sub(r"(?:\*\*|__|`)", "", lines[-1]).strip()
    match = _TERMINAL_ANSWER_FAMILY.search(normalized)
    if match is None:
        return FormatResult(False, "missing_terminal_answer_statement")
    number = match.group("number_colon") or match.group("number_statement")
    return FormatResult(True, "ok", _decimal(number))


def format_results(text: str) -> dict[str, FormatResult]:
    strict = parse_generated_answer(text)
    return {
        "frozen_strict_standalone_line": FormatResult(
            strict.ok, strict.status, strict.value
        ),
        "terminal_final_marker_suffix": terminal_final_marker(text),
        "terminal_explicit_answer_statement": terminal_answer_statement(text),
    }


def failure_taxonomy(text: str, criteria: dict[str, FormatResult] | None = None) -> str:
    """Assign one deterministic, exclusive category to a strict-format failure."""

    criteria = criteria or format_results(text)
    if criteria["frozen_strict_standalone_line"].ok:
        return "strict_ok"
    if not text.strip():
        return "empty_output"
    markers = list(_FINAL_MARKER.finditer(text))
    if len(markers) > 1:
        return "multiple_final_markers"
    if not markers:
        if criteria["terminal_explicit_answer_statement"].ok:
            return "alternate_terminal_answer_statement"
        if _ANY_NUMBER.search(text):
            return "missing_final_marker_with_numeric"
        return "missing_final_marker_no_numeric"

    payload = text[markers[0].end() :]
    number = _PREFIX_NUMBER.match(payload)
    if number is None:
        return "invalid_final_payload"
    remainder = payload[number.end() :]
    if remainder and not _ONLY_DECORATION.fullmatch(remainder):
        return "extra_text_after_final_numeric"

    marker_line_index = text[: markers[0].start()].count("\n")
    nonempty_indices = [
        index for index, line in enumerate(text.splitlines()) if line.strip()
    ]
    if nonempty_indices and marker_line_index != nonempty_indices[-1]:
        return "marker_not_final"
    if criteria["terminal_final_marker_suffix"].ok:
        return "embedded_or_decorated_marker_on_final_line"
    return "marker_or_payload_not_terminal"


def recompute_frozen_row(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute the frozen strict+fallback scoring path for one raw output."""

    text = str(row.get("raw_output", ""))
    strict = parse_generated_answer(text)
    if strict.ok:
        prediction = strict
        parse_mode = "strict_final_marker"
    else:
        prediction = parse_last_numeric_answer(text)
        parse_mode = "last_numeric_fallback" if prediction.ok else "failed"
    gold = _decimal(row.get("gold_value"))
    numeric_correct = bool(
        prediction.ok and gold is not None and prediction.value == gold
    )
    return {
        "strict_parse_status": strict.status,
        "strict_parsed_prediction": (
            str(strict.value) if strict.value is not None else None
        ),
        "parse_mode": parse_mode,
        "parse_status": prediction.status,
        "parsed_prediction": (
            str(prediction.value) if prediction.value is not None else None
        ),
        "numeric_correct": numeric_correct,
    }


def parser_mismatches(row: dict[str, Any]) -> list[str]:
    recomputed = recompute_frozen_row(row)
    mismatches: list[str] = []
    for key in ("strict_parse_status", "parse_mode", "parse_status", "numeric_correct"):
        if key in row and row[key] != recomputed[key]:
            mismatches.append(key)
    for key in ("strict_parsed_prediction", "parsed_prediction"):
        if key in row and not _same_decimal(row[key], recomputed[key]):
            mismatches.append(key)
    return mismatches


def _discover_sources(inputs: Iterable[Path]) -> list[Path]:
    sources: dict[str, Path] = {}
    for raw in inputs:
        path = raw.resolve()
        if path.is_file():
            if path.name.endswith(".tar.gz"):
                sources[str(path).lower()] = path
            else:
                raise ValueError(f"unsupported evidence file: {path}")
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        if (path / "manifest.json").is_file() and (
            path / DATASET_MEMBERS["gsm8k"]
        ).is_file():
            sources[str(path).lower()] = path
            continue
        archives = sorted(path.rglob("*.tar.gz"))
        if archives:
            for archive in archives:
                sources[str(archive.resolve()).lower()] = archive.resolve()
            continue
        for manifest in path.rglob("manifest.json"):
            cell_root = manifest.parent
            if (cell_root / DATASET_MEMBERS["gsm8k"]).is_file():
                sources[str(cell_root.resolve()).lower()] = cell_root.resolve()
    if not sources:
        raise ValueError("no evidence archives or extracted cell artifacts found")
    return sorted(sources.values(), key=lambda item: str(item).lower())


def _load_cell(source: Path) -> CellData:
    with _open_reader(source) as reader:
        required = [
            "manifest.json",
            "training_complete/token_audit.json",
            "training_complete/token_budget_audit.json",
            "audit/formal_cell_audit.json",
            "audit/ood_audit.json",
            *DATASET_MEMBERS.values(),
        ]
        missing = [member for member in required if not reader.exists(member)]
        if missing:
            raise ValueError(f"{source} is missing evidence members: {missing}")
        manifest = _json(reader, "manifest.json")
        formal_audit = _json(reader, "audit/formal_cell_audit.json")
        ood_audit = _json(reader, "audit/ood_audit.json")
        if str(formal_audit.get("status", formal_audit.get("verdict", ""))).upper() != "PASS":
            raise ValueError(f"formal audit is not PASS: {source}")
        if str(ood_audit.get("status", ood_audit.get("verdict", ""))).upper() != "PASS":
            raise ValueError(f"OOD audit is not PASS: {source}")
        config = manifest["config"]
        outputs = {
            dataset: _jsonl(reader, member)
            for dataset, member in DATASET_MEMBERS.items()
        }
        token_audit = _json(reader, "training_complete/token_audit.json")
        token_budget = _json(
            reader, "training_complete/token_budget_audit.json"
        )
    source_sha256 = (
        _sha256_file(source)
        if source.is_file()
        else _directory_input_sha256(source, required)
    )
    return CellData(
        cell_id=str(config["cell_id"]),
        method=str(config["method"]),
        replicate_index=int(config["replicate_index"]),
        train_seed=int(manifest["seed"]),
        manifest=manifest,
        token_audit=token_audit,
        token_budget_audit=token_budget,
        outputs=outputs,
        source_path=source,
        source_sha256=source_sha256,
    )


def _selection_index(root: Path | None) -> dict[str, tuple[Path, dict[str, Any]]]:
    if root is None:
        return {}
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.resolve().rglob("selection_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_hash = _sha256_file(path)
        if file_hash in index:
            raise ValueError(f"duplicate selection manifest SHA-256: {file_hash}")
        index[file_hash] = (path, payload)
    return index


def _duplicate_mapping(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = str(row["candidate_id"])
            if candidate_id in mapping:
                raise ValueError(f"duplicate candidate in cluster map: {candidate_id}")
            mapping[candidate_id] = str(row["near_duplicate_cluster_id"])
    return mapping


def _response_band(value: int) -> str:
    if value <= 16:
        return "000-016"
    if value <= 32:
        return "017-032"
    if value <= 64:
        return "033-064"
    if value <= 128:
        return "065-128"
    if value <= 256:
        return "129-256"
    return "257-plus"


def _total_band(value: int) -> str:
    if value <= 128:
        return "000-128"
    if value <= 256:
        return "129-256"
    if value <= 384:
        return "257-384"
    if value <= 511:
        return "385-511"
    return "512"


def _generated_token_count(row: dict[str, Any]) -> int | None:
    for key in (
        "generated_token_count",
        "generation_token_count",
        "new_token_count",
    ):
        value = row.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _criterion_value_correct(result: FormatResult, gold: Any) -> bool:
    gold_value = _decimal(gold)
    return bool(result.ok and gold_value is not None and result.value == gold_value)


def _counter_rows(
    *,
    cell: CellData,
    dimension: str,
    values: Iterable[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    aggregate: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for category, supervised, total in values:
        row = aggregate[category]
        row[0] += 1
        row[1] += supervised
        row[2] += total
        row[3] += max(0, total - supervised)
    return [
        {
            "cell_id": cell.cell_id,
            "method": cell.method,
            "replicate_index": cell.replicate_index,
            "dimension": dimension,
            "category": category,
            "count": counts[0],
            "supervised_tokens": counts[1],
            "total_tokens": counts[2],
            "prompt_tokens": counts[3],
        }
        for category, counts in sorted(aggregate.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _descriptive_gate(
    criterion_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    methods = {"random_free_mix", "rds_error_free_mix"}
    relaxed = {
        "terminal_final_marker_suffix",
        "terminal_explicit_answer_statement",
    }
    ood = {"svamp", "asdiv_numeric", "multiarith"}
    rates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in criterion_rows:
        if (
            row["method"] in methods
            and row["criterion"] in relaxed
            and row["dataset"] in ood
        ):
            rates[(row["method"], row["criterion"], row["dataset"])].append(
                float(row["format_rate"])
            )
    differences: list[dict[str, Any]] = []
    for criterion in sorted(relaxed):
        for dataset in sorted(ood):
            left = rates.get(("rds_error_free_mix", criterion, dataset), [])
            right = rates.get(("random_free_mix", criterion, dataset), [])
            if not left or not right:
                continue
            left_mean = sum(left) / len(left)
            right_mean = sum(right) / len(right)
            differences.append(
                {
                    "criterion": criterion,
                    "dataset": dataset,
                    "rds_error_free_mean": left_mean,
                    "random_free_mean": right_mean,
                    "difference": left_mean - right_mean,
                    "difference_pp": 100.0 * (left_mean - right_mean),
                    "descriptive_only": True,
                }
            )
    tasks_by_criterion = Counter()
    for row in differences:
        if row["difference"] <= -0.10:
            tasks_by_criterion[row["criterion"]] += 1
    criteria_passing = [
        name for name, count in tasks_by_criterion.items() if count >= 2
    ]
    return {
        "status": (
            "descriptive_upgrade_signal"
            if len(criteria_passing) >= 2
            else "engineering_diagnostic_only_signal"
        ),
        "criteria_with_at_least_two_tasks_below_minus_10pp": sorted(
            criteria_passing
        ),
        "differences": differences,
        "interpretation_limit": (
            "Descriptive sensitivity check only; selection policies are not "
            "naturally paired and this is not a causal or confirmatory test."
        ),
    }


def _render_cn_report(summary: dict[str, Any]) -> str:
    gate = summary["format_sensitivity_gate"]
    lines = [
        "# Phase 1A CPU-only 格式与数据组成审计",
        "",
        f"- 审计 cell：{summary['cell_count']}",
        f"- 原始输出：{summary['raw_output_count']}",
        f"- 冻结 parser 逐条重算不一致：{summary['parser_mismatch_count']}",
        f"- 确定性失败样本：{summary['failure_sample_count']}",
        f"- 格式敏感性信号：`{gate['status']}`",
        "",
        "## 重要边界",
        "",
        "1. 冻结 strict parser 和原正式指标没有被修改。",
        "2. 两种放宽格式判据只是敏感性分析，不替代主指标。",
        "3. common/free 及 RDS/random 不自动构成因果中介分解或天然配对。",
        "4. 格式敏感性 gate 只是描述性信号，不是确证性结论。",
        "",
        "## 产物",
        "",
        "- `summary.json`：机器可读总结与 provenance。",
        "- `cell_summary.csv`：每格 token 预算和暴露量。",
        "- `selection_composition.csv`：source、response/prompt/total token band 组成。",
        "- `format_criteria.csv`：三种判据的 strict×numeric 四象限。",
        "- `generation_lengths.jsonl`：逐输出字符长度、行数与可用的 generated-token 计数。",
        "- `generation_length_summary.csv`：cell×dataset×格式×numeric 四象限长度摘要。",
        "- `failure_taxonomy.csv`：互斥的格式失败类型。",
        "- `failure_sample_100.jsonl`：基于 SHA-256 排序的确定性样本。",
        "- `duplicate_summary.csv`：近重复簇映射覆盖与碰撞摘要。",
        "- `estimand_note.md`：固定政策、policy expectation 与因果边界。",
        "",
    ]
    return "\n".join(lines)


def run_cpu_identifiability_audit(
    *,
    inputs: Iterable[Path],
    output_root: Path,
    selection_root: Path | None = None,
    near_duplicate_clusters: Path | None = None,
    expected_cells: int | None = 16,
    expected_dataset_counts: bool = True,
    sample_size: int = 100,
    sample_seed: int = 20260827,
    run_id: str | None = None,
) -> Path:
    """Run the read-only audit and create one non-overwriting artifact directory."""

    sources = _discover_sources(inputs)
    cells = [_load_cell(source) for source in sources]
    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        duplicates = sorted(
            cell_id for cell_id, count in Counter(cell_ids).items() if count > 1
        )
        raise ValueError(f"duplicate evidence cells: {duplicates}")
    if expected_cells is not None and len(cells) != expected_cells:
        raise ValueError(f"expected {expected_cells} cells, found {len(cells)}")

    selection_index = _selection_index(selection_root)
    for cell in cells:
        expected_hash = str(cell.manifest["config"]["selection_manifest_sha256"])
        selected = selection_index.get(expected_hash)
        if selected is not None:
            cell.selection_manifest = selected[1]
            if str(selected[1]["selected_id_sha256"]) != str(
                cell.manifest["config"]["selected_id_sha256"]
            ):
                raise ValueError(f"selected ID SHA mismatch for {cell.cell_id}")

    duplicate_map = _duplicate_mapping(near_duplicate_clusters)
    parser_mismatch_rows: list[dict[str, Any]] = []
    format_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    sample_candidates: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    generation_groups: dict[
        tuple[str, str, int, str, str, bool, bool, str],
        list[tuple[int, int, int]],
    ] = defaultdict(list)
    raw_output_count = 0

    for cell in sorted(cells, key=lambda item: item.cell_id):
        token_by_id = {str(row["candidate_id"]): row for row in cell.token_audit}
        unique_tokens = sum(int(row["supervised_tokens"]) for row in cell.token_audit)
        exposure = int(
            cell.token_budget_audit["response_supervision_exposure_tokens"]
        )
        cell_rows.append(
            {
                "cell_id": cell.cell_id,
                "method": cell.method,
                "replicate_index": cell.replicate_index,
                "train_seed": cell.train_seed,
                "selected_count": len(cell.token_audit),
                "unique_response_supervision_tokens": unique_tokens,
                "response_supervision_exposure_tokens": exposure,
                "optimizer_steps": int(cell.token_budget_audit["optimizer_steps"]),
                "selection_manifest_bound": cell.selection_manifest is not None,
                "source_artifact": _public_artifact_label(cell.source_path),
                "source_sha256": cell.source_sha256,
            }
        )

        candidates = (
            list(cell.selection_manifest["selected_candidates"])
            if cell.selection_manifest is not None
            else [
                {
                    **row,
                    "source_dataset": "unknown_without_selection_manifest",
                    "common_mix_stratum": "unknown_without_selection_manifest",
                }
                for row in cell.token_audit
            ]
        )
        if set(token_by_id) != {str(row["candidate_id"]) for row in candidates}:
            raise ValueError(f"selection/token audit candidate mismatch: {cell.cell_id}")
        for candidate in candidates:
            audit = token_by_id[str(candidate["candidate_id"])]
            for key in ("supervised_tokens", "total_tokens"):
                if int(candidate[key]) != int(audit[key]):
                    raise ValueError(f"{key} mismatch in {cell.cell_id}")
        composition_rows.extend(
            _counter_rows(
                cell=cell,
                dimension="source_dataset",
                values=(
                    (
                        str(row.get("source_dataset", "unknown")),
                        int(row["supervised_tokens"]),
                        int(row["total_tokens"]),
                    )
                    for row in candidates
                ),
            )
        )
        composition_rows.extend(
            _counter_rows(
                cell=cell,
                dimension="prompt_token_band",
                values=(
                    (
                        _total_band(
                            max(
                                0,
                                int(row["total_tokens"])
                                - int(row["supervised_tokens"]),
                            )
                        ),
                        int(row["supervised_tokens"]),
                        int(row["total_tokens"]),
                    )
                    for row in candidates
                ),
            )
        )
        composition_rows.extend(
            _counter_rows(
                cell=cell,
                dimension="response_token_band",
                values=(
                    (
                        _response_band(int(row["supervised_tokens"])),
                        int(row["supervised_tokens"]),
                        int(row["total_tokens"]),
                    )
                    for row in candidates
                ),
            )
        )
        composition_rows.extend(
            _counter_rows(
                cell=cell,
                dimension="total_token_band",
                values=(
                    (
                        _total_band(int(row["total_tokens"])),
                        int(row["supervised_tokens"]),
                        int(row["total_tokens"]),
                    )
                    for row in candidates
                ),
            )
        )

        selected_clusters = [
            duplicate_map.get(str(row["candidate_id"])) for row in candidates
        ]
        covered_clusters = [cluster for cluster in selected_clusters if cluster]
        cluster_counts = Counter(covered_clusters)
        duplicate_rows.append(
            {
                "cell_id": cell.cell_id,
                "method": cell.method,
                "replicate_index": cell.replicate_index,
                "selected_count": len(candidates),
                "cluster_mapping_covered": len(covered_clusters),
                "cluster_mapping_missing": len(candidates) - len(covered_clusters),
                "unique_selected_clusters": len(cluster_counts),
                "multi_selected_cluster_count": sum(
                    count > 1 for count in cluster_counts.values()
                ),
                "cluster_collision_excess": sum(
                    max(0, count - 1) for count in cluster_counts.values()
                ),
                "maximum_selected_per_cluster": max(cluster_counts.values(), default=0),
            }
        )

        for dataset, rows in cell.outputs.items():
            if expected_dataset_counts and len(rows) != EXPECTED_DATASET_COUNTS[dataset]:
                raise ValueError(
                    f"{cell.cell_id}/{dataset}: expected "
                    f"{EXPECTED_DATASET_COUNTS[dataset]} rows, found {len(rows)}"
                )
            record_ids = [str(row["record_id"]) for row in rows]
            if len(record_ids) != len(set(record_ids)):
                raise ValueError(f"duplicate record IDs: {cell.cell_id}/{dataset}")
            raw_output_count += len(rows)
            criterion_counts: dict[str, Counter[tuple[bool, bool]]] = defaultdict(Counter)
            taxonomy_count: Counter[str] = Counter()
            for row in rows:
                mismatches = parser_mismatches(row)
                if mismatches:
                    parser_mismatch_rows.append(
                        {
                            "cell_id": cell.cell_id,
                            "dataset": dataset,
                            "record_id": row.get("record_id"),
                            "fields": mismatches,
                        }
                    )
                raw_output = str(row.get("raw_output", ""))
                criteria = format_results(raw_output)
                numeric_correct = bool(row.get("numeric_correct", False))
                char_length = len(raw_output)
                line_count = len(raw_output.splitlines())
                generated_tokens = _generated_token_count(row)
                length_measure = (
                    "generated_token_count"
                    if generated_tokens is not None
                    else "generation_char_length_proxy"
                )
                primary_length = (
                    generated_tokens if generated_tokens is not None else char_length
                )
                generation_rows.append(
                    {
                        "cell_id": cell.cell_id,
                        "method": cell.method,
                        "replicate_index": cell.replicate_index,
                        "dataset": dataset,
                        "record_id": row.get("record_id"),
                        "numeric_correct": numeric_correct,
                        "frozen_strict_ok": criteria[
                            "frozen_strict_standalone_line"
                        ].ok,
                        "terminal_final_marker_suffix_ok": criteria[
                            "terminal_final_marker_suffix"
                        ].ok,
                        "terminal_explicit_answer_statement_ok": criteria[
                            "terminal_explicit_answer_statement"
                        ].ok,
                        "generation_char_length": char_length,
                        "generation_line_count": line_count,
                        "generated_token_count": generated_tokens,
                        "primary_length_measure": length_measure,
                        "primary_length_value": primary_length,
                    }
                )
                for criterion, result in criteria.items():
                    criterion_counts[criterion][(result.ok, numeric_correct)] += 1
                    generation_groups[
                        (
                            cell.cell_id,
                            cell.method,
                            cell.replicate_index,
                            dataset,
                            criterion,
                            result.ok,
                            numeric_correct,
                            length_measure,
                        )
                    ].append((char_length, line_count, primary_length))
                taxonomy = failure_taxonomy(
                    raw_output, criteria
                )
                taxonomy_count[taxonomy] += 1
                if taxonomy != "strict_ok":
                    sample_candidates.append(
                        {
                            "cell_id": cell.cell_id,
                            "method": cell.method,
                            "replicate_index": cell.replicate_index,
                            "dataset": dataset,
                            "record_id": row.get("record_id"),
                            "gold_value": row.get("gold_value"),
                            "numeric_correct": numeric_correct,
                            "stored_strict_parse_status": row.get(
                                "strict_parse_status"
                            ),
                            "taxonomy": taxonomy,
                            "criteria": {
                                name: {
                                    "ok": result.ok,
                                    "status": result.status,
                                    "parsed_value": (
                                        str(result.value)
                                        if result.value is not None
                                        else None
                                    ),
                                    "matches_gold": _criterion_value_correct(
                                        result, row.get("gold_value")
                                    ),
                                }
                                for name, result in criteria.items()
                            },
                            "raw_output": row.get("raw_output", ""),
                        }
                    )
            for criterion, counts in sorted(criterion_counts.items()):
                total = len(rows)
                strict_true = sum(
                    count for (format_ok, _), count in counts.items() if format_ok
                )
                format_rows.append(
                    {
                        "cell_id": cell.cell_id,
                        "method": cell.method,
                        "replicate_index": cell.replicate_index,
                        "dataset": dataset,
                        "criterion": criterion,
                        "total": total,
                        "format_ok_numeric_correct": counts[(True, True)],
                        "format_ok_numeric_wrong": counts[(True, False)],
                        "format_fail_numeric_correct": counts[(False, True)],
                        "format_fail_numeric_wrong": counts[(False, False)],
                        "format_rate": strict_true / total if total else 0.0,
                        "numeric_accuracy": (
                            sum(
                                count
                                for (_, numeric), count in counts.items()
                                if numeric
                            )
                            / total
                            if total
                            else 0.0
                        ),
                    }
                )
            for taxonomy, count in sorted(taxonomy_count.items()):
                taxonomy_rows.append(
                    {
                        "cell_id": cell.cell_id,
                        "method": cell.method,
                        "replicate_index": cell.replicate_index,
                        "dataset": dataset,
                        "taxonomy": taxonomy,
                        "count": count,
                        "rate": count / len(rows) if rows else 0.0,
                    }
                )

    if parser_mismatch_rows:
        examples = parser_mismatch_rows[:10]
        raise ValueError(
            f"frozen parser recomputation mismatch in {len(parser_mismatch_rows)} rows: "
            f"{examples}"
        )

    generation_summary_rows: list[dict[str, Any]] = []
    for key, values in sorted(generation_groups.items()):
        (
            cell_id,
            method,
            replicate_index,
            dataset,
            criterion,
            format_ok,
            numeric_correct,
            length_measure,
        ) = key
        char_lengths = [value[0] for value in values]
        line_counts = [value[1] for value in values]
        primary_lengths = [value[2] for value in values]
        generation_summary_rows.append(
            {
                "cell_id": cell_id,
                "method": method,
                "replicate_index": replicate_index,
                "dataset": dataset,
                "criterion": criterion,
                "format_ok": format_ok,
                "numeric_correct": numeric_correct,
                "primary_length_measure": length_measure,
                "count": len(values),
                "generation_char_length_mean": mean(char_lengths),
                "generation_char_length_median": median(char_lengths),
                "generation_char_length_p90": _percentile(char_lengths, 0.90),
                "generation_line_count_mean": mean(line_counts),
                "generation_line_count_median": median(line_counts),
                "generation_line_count_p90": _percentile(line_counts, 0.90),
                "primary_length_mean": mean(primary_lengths),
                "primary_length_median": median(primary_lengths),
                "primary_length_p90": _percentile(primary_lengths, 0.90),
            }
        )

    ranked_samples = sorted(
        sample_candidates,
        key=lambda row: hashlib.sha256(
            (
                f"{sample_seed}|{row['cell_id']}|{row['dataset']}|"
                f"{row['record_id']}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    selected_samples = ranked_samples[:sample_size]
    gate = _descriptive_gate(format_rows)

    input_fingerprint = hashlib.sha256(
        "\n".join(
            f"{cell.cell_id}:{cell.source_sha256 or cell.source_path}"
            for cell in sorted(cells, key=lambda item: item.cell_id)
        ).encode("utf-8")
    ).hexdigest()
    if run_id is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_cpu_identifiability_{input_fingerprint[:10]}"
    output_root = output_root.resolve()
    output_dir = output_root / run_id
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "cell_count": len(cells),
        "raw_output_count": raw_output_count,
        "parser_mismatch_count": len(parser_mismatch_rows),
        "failure_sample_count": len(selected_samples),
        "failure_sample_requested": sample_size,
        "sample_seed": sample_seed,
        "input_fingerprint_sha256": input_fingerprint,
        "selection_manifest_binding_count": sum(
            cell.selection_manifest is not None for cell in cells
        ),
        "near_duplicate_mapping_supplied": bool(duplicate_map),
        "generation_length_policy": {
            "preferred": "generated token count stored in each raw row",
            "fallback": "Unicode character count proxy when token count is absent",
            "line_count": "Python splitlines count",
            "aggregation": "cell x dataset x criterion x format_ok x numeric_correct",
        },
        "analysis_code": {
            "cpu_identifiability_audit_sha256": _sha256_file(Path(__file__)),
            "frozen_parser_sha256": _sha256_file(
                Path(frozen_parser_module.__file__)
            ),
        },
        "near_duplicate_clusters": (
            {
                "artifact": near_duplicate_clusters.name,
                "sha256": _sha256_file(near_duplicate_clusters.resolve()),
            }
            if near_duplicate_clusters is not None
            else None
        ),
        "format_criteria": [
            "frozen_strict_standalone_line",
            "terminal_final_marker_suffix",
            "terminal_explicit_answer_statement",
        ],
        "format_sensitivity_gate": gate,
        "claim_boundary": (
            "All relaxed format results are descriptive sensitivity analyses; "
            "they do not replace the frozen primary parser or identify causal effects."
        ),
        "inputs": [
            {
                "cell_id": cell.cell_id,
                "artifact": _public_artifact_label(cell.source_path),
                "sha256": cell.source_sha256,
                "selection_manifest_sha256": cell.manifest["config"][
                    "selection_manifest_sha256"
                ],
            }
            for cell in sorted(cells, key=lambda item: item.cell_id)
        ],
    }

    _write_json(output_dir / "summary.json", summary)
    _write_csv(
        output_dir / "cell_summary.csv",
        cell_rows,
        [
            "cell_id",
            "method",
            "replicate_index",
            "train_seed",
            "selected_count",
            "unique_response_supervision_tokens",
            "response_supervision_exposure_tokens",
            "optimizer_steps",
            "selection_manifest_bound",
            "source_artifact",
            "source_sha256",
        ],
    )
    _write_csv(
        output_dir / "selection_composition.csv",
        composition_rows,
        [
            "cell_id",
            "method",
            "replicate_index",
            "dimension",
            "category",
            "count",
            "supervised_tokens",
            "total_tokens",
            "prompt_tokens",
        ],
    )
    _write_csv(
        output_dir / "format_criteria.csv",
        format_rows,
        [
            "cell_id",
            "method",
            "replicate_index",
            "dataset",
            "criterion",
            "total",
            "format_ok_numeric_correct",
            "format_ok_numeric_wrong",
            "format_fail_numeric_correct",
            "format_fail_numeric_wrong",
            "format_rate",
            "numeric_accuracy",
        ],
    )
    _write_csv(
        output_dir / "failure_taxonomy.csv",
        taxonomy_rows,
        [
            "cell_id",
            "method",
            "replicate_index",
            "dataset",
            "taxonomy",
            "count",
            "rate",
        ],
    )
    _write_csv(
        output_dir / "duplicate_summary.csv",
        duplicate_rows,
        [
            "cell_id",
            "method",
            "replicate_index",
            "selected_count",
            "cluster_mapping_covered",
            "cluster_mapping_missing",
            "unique_selected_clusters",
            "multi_selected_cluster_count",
            "cluster_collision_excess",
            "maximum_selected_per_cluster",
        ],
    )
    _write_jsonl(output_dir / "generation_lengths.jsonl", generation_rows)
    _write_csv(
        output_dir / "generation_length_summary.csv",
        generation_summary_rows,
        [
            "cell_id",
            "method",
            "replicate_index",
            "dataset",
            "criterion",
            "format_ok",
            "numeric_correct",
            "primary_length_measure",
            "count",
            "generation_char_length_mean",
            "generation_char_length_median",
            "generation_char_length_p90",
            "generation_line_count_mean",
            "generation_line_count_median",
            "generation_line_count_p90",
            "primary_length_mean",
            "primary_length_median",
            "primary_length_p90",
        ],
    )
    _write_jsonl(output_dir / "failure_sample_100.jsonl", selected_samples)
    (output_dir / "estimand_note.md").write_text(
        ESTIMAND_NOTE, encoding="utf-8"
    )
    (output_dir / "report_cn.md").write_text(
        _render_cn_report(summary), encoding="utf-8"
    )
    file_rows = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file():
            file_rows.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    _write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "files": file_rows,
        },
    )
    return output_dir
