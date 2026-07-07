from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from eg_sft.utils.io import read_jsonl  # noqa: E402
from eg_sft.eval.metrics import numeric_equal  # noqa: E402

NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?", re.IGNORECASE)
AUDIT_CATEGORIES = [
    "parser_tail_fragment_risk",
    "parse_failure",
    "percent_change_calculation_error",
    "weighted_formula_error",
    "temporal_calculation_error",
    "multiplication_calculation_error",
    "direct_wrong_number_or_relation_error",
]


def numeric_tokens(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def numeric_values(text: str) -> list[float]:
    values = []
    for token in numeric_tokens(text):
        try:
            values.append(float(token.rstrip("%")))
        except ValueError:
            continue
    return values


def clean_problem_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if prompt.startswith("Problem: "):
        prompt = prompt[len("Problem: ") :]
    marker = "\nFinal numeric answer ="
    if marker in prompt:
        prompt = prompt.split(marker, 1)[0]
    return prompt.strip()


def translate_problem_prompt(prompt: str) -> str:
    problem = clean_problem_prompt(prompt)
    ratio = re.fullmatch(
        r"A metric starts at ([\d.]+) and then has a ([\d.]+)% (increase|decrease)\. "
        r"What is the final value\?",
        problem,
    )
    if ratio:
        direction = "上升" if ratio.group(3) == "increase" else "下降"
        return f"一个指标从 {ratio.group(1)} 开始，随后{direction} {ratio.group(2)}%。最终值是多少？"

    multiply = re.fullmatch(r"Multiply the related counts (.+)\. What is the product\?", problem)
    if multiply:
        counts = multiply.group(1).replace(", ", "、")
        return f"将相关数量 {counts} 相乘，乘积是多少？"

    temporal = re.fullmatch(
        r"Start from ([\d.]+) and apply ordered changes \[(.+)\]\. What is the final value\?",
        problem,
    )
    if temporal:
        changes = temporal.group(2).replace(", ", "、")
        return f"从 {temporal.group(1)} 开始，按顺序应用变化 [{changes}]。最终值是多少？"

    weighted = re.fullmatch(r"A weighted metric uses (.+)\. What is the weighted aggregate\?", problem)
    if weighted:
        return f"一个加权指标使用这些权重和值：{weighted.group(1)}。加权聚合值是多少？"

    return problem


def write_csv_for_spreadsheets(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_titled_csv_for_spreadsheets(path: Path, title: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([title])
        writer.writerow([])
        if not rows:
            return
        writer.writerow(list(rows[0].keys()))
        for row in rows:
            writer.writerow([row[key] for key in rows[0].keys()])


def audit_category(row: dict) -> str:
    text = str(row.get("raw_continuation", ""))
    if bool(row.get("numeric_accuracy")):
        return "correct_output"
    if not bool(row.get("parse_success")):
        return "parse_failure"

    answer = float(row["answer"])
    parsed = row.get("parsed_prediction")
    values = numeric_values(text)
    if parsed not in {"", None} and any(numeric_equal(value, answer) for value in values):
        return "parser_tail_fragment_risk"

    family = row.get("task_family")
    if family == "ratio_change":
        return "percent_change_calculation_error"
    if family == "weighted_aggregation":
        return "weighted_formula_error"
    if family == "temporal_numeric_constraint":
        return "temporal_calculation_error"
    if family == "multiplicative_relation":
        return "multiplication_calculation_error"

    return "direct_wrong_number_or_relation_error"


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        category = audit_category(row)
        key = (row["task_family"], row["difficulty_bucket"])
        grouped[key]["count"] += 1
        grouped[key]["correct_count"] += int(bool(row["numeric_accuracy"]))
        grouped[key]["errors"] += int(not bool(row["numeric_accuracy"]))
        grouped[key][category] += 1
        grouped[key]["multi_number_outputs"] += int(len(numeric_tokens(row["raw_continuation"])) >= 3)
        grouped[key]["equation_outputs"] += int("=" in row["raw_continuation"])

    output = []
    for (family, difficulty), stats in sorted(grouped.items()):
        count = stats["count"]
        errors = stats["errors"]
        output.append(
            {
                "task_family": family,
                "difficulty_bucket": difficulty,
                "count": count,
                "accuracy": round(stats["correct_count"] / count, 6) if count else 0,
                "error_rate": round(errors / count, 6) if count else 0,
                **{category: stats[category] for category in AUDIT_CATEGORIES},
                "multi_number_output_rate": round(stats["multi_number_outputs"] / count, 6)
                if count
                else 0,
                "equation_output_rate": round(stats["equation_outputs"] / count, 6)
                if count
                else 0,
            }
        )
    return output


def representative_examples(rows: list[dict], limit_per_category: int) -> list[dict]:
    selected: list[dict] = []
    counts: Counter = Counter()
    for row in rows:
        category = audit_category(row)
        if category == "correct_output" or counts[category] >= limit_per_category:
            continue
        counts[category] += 1
        selected.append(
            {
                "id": row["id"],
                "prompt": clean_problem_prompt(row["prompt"]),
                "prompt_zh": translate_problem_prompt(row["prompt"]),
                "task_family": row["task_family"],
                "difficulty_bucket": row["difficulty_bucket"],
                "answer": row["answer"],
                "parsed_prediction": row["parsed_prediction"],
                "audit_category": category,
                "number_token_count": len(numeric_tokens(row["raw_continuation"])),
                "has_equation": "=" in row["raw_continuation"],
                "human_check_prompt": "请判断这是真推理错误、parser/格式问题，还是题目理解问题。",
                "raw_continuation": row["raw_continuation"].replace("\n", " | "),
            }
        )
    return selected


def chinese_review_examples(rows: list[dict]) -> list[dict]:
    category_zh = {
        "parser_tail_fragment_risk": "parser 尾部残片风险",
        "parse_failure": "解析失败",
        "percent_change_calculation_error": "百分比变化计算错误",
        "weighted_formula_error": "加权公式/聚合方式错误",
        "temporal_calculation_error": "时间顺序加减计算错误",
        "multiplication_calculation_error": "乘法计算错误",
        "direct_wrong_number_or_relation_error": "直接数值或关系理解错误",
    }
    family_zh = {
        "multiplicative_relation": "乘法关系",
        "ratio_change": "比例变化",
        "temporal_numeric_constraint": "时间约束数值题",
        "weighted_aggregation": "加权聚合",
    }
    difficulty_zh = {"easy": "简单", "medium": "中等", "hard": "困难"}
    output = []
    for row in rows:
        output.append(
            {
                "样例编号": row["id"],
                "题目中文说明": row["prompt_zh"],
                "原始题目（英文）": row["prompt"],
                "任务类型": family_zh.get(row["task_family"], row["task_family"]),
                "难度": difficulty_zh.get(row["difficulty_bucket"], row["difficulty_bucket"]),
                "标准答案": row["answer"],
                "解析出的模型答案": row["parsed_prediction"],
                "自动初判类型": category_zh.get(row["audit_category"], row["audit_category"]),
                "输出中的数字个数": row["number_token_count"],
                "是否包含等式": "是" if row["has_equation"] else "否",
                "请你人工判断": "这是真推理错误、parser/格式问题，还是题目理解问题？",
                "模型原始输出": row["raw_continuation"],
                "人工复核结论（你填写）": "",
                "备注（你填写）": "",
            }
        )
    return output


def write_research_note(path: Path, rows: list[dict], summary: list[dict], examples: list[dict]) -> None:
    total = len(rows)
    correct = sum(bool(row["numeric_accuracy"]) for row in rows)
    errors = total - correct
    category_counts = Counter(audit_category(row) for row in rows)
    family_rows = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in rows:
        family_rows[row["task_family"]]["n"] += 1
        family_rows[row["task_family"]]["correct"] += int(bool(row["numeric_accuracy"]))

    family_lines = []
    for family, stats in sorted(family_rows.items()):
        acc = stats["correct"] / stats["n"] if stats["n"] else 0
        family_lines.append(f"- `{family}`: {stats['correct']}/{stats['n']} correct, accuracy={acc:.2f}")

    example_lines = []
    for row in examples[:12]:
        example_lines.append(
            "- "
            f"`{row['id']}` ({row['task_family']}, {row['difficulty_bucket']}): "
            f"answer={row['answer']}, parsed={row['parsed_prediction']}, "
            f"category={row['audit_category']}; output: {row['raw_continuation']}"
        )

    category_lines = []
    category_zh = {
        "parser_tail_fragment_risk": "parser 尾部残片风险",
        "parse_failure": "解析失败",
        "percent_change_calculation_error": "百分比变化计算错误",
        "weighted_formula_error": "加权公式/聚合方式错误",
        "temporal_calculation_error": "时间顺序加减计算错误",
        "multiplication_calculation_error": "乘法计算错误",
        "direct_wrong_number_or_relation_error": "直接数值或关系理解错误",
    }
    for category in AUDIT_CATEGORIES:
        count = category_counts[category]
        if count:
            category_lines.append(f"- `{category}`（{category_zh[category]}）: {count} 条。")

    note = f"""# 真实错误画像第一轮研究笔记

## 当前观察

本轮只分析 `Qwen/Qwen2.5-0.5B` 在 `dev_diagnostic` 上的真实 base diagnostic。样本数为 {total}，正确 {correct}，错误 {errors}，数值准确率为 {correct / total:.2f}。这不是 LoRA 结果，也不是 Targeted selection 优于 Random 的证据。

按任务族粗看：

{chr(10).join(family_lines)}

## 错误类型的初步判断

下面分类是自动启发式审计，不是人工标注真值。它的作用是帮助挑选需要人工复核的样例，而不是直接定义最终错误类型。

{chr(10).join(category_lines)}

## 代表样例

{chr(10).join(example_lines)}

## 对研究问题的影响

这轮结果提示 error-guided selection 值得继续分析，但不能直接进入方法有效性声明。真实错误画像里至少混合了三种信号：模型确实不会算、输出格式导致 parser 可能误判、以及题目措辞可能诱导模型用错公式。下一步应先让人工看 20-30 条错误样例，确认哪些错误适合用 SFT 数据修复，哪些应先改 prompt/parser。

## 你的参与方式

你在这个阶段最适合做研究负责人，而不是数据提供者。建议你直接打开 `results/real_parser_audit_examples_cn_with_prompts.csv`，按中文表头逐条判断：这是真推理错误、格式/parser 风险，还是题目理解问题。你的判断会决定下一步是改 parser、微调 prompt，还是进入 selection bias audit。
"""
    path.write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/real_base_diagnostic_outputs.jsonl")
    parser.add_argument("--summary", default="results/real_parser_audit_summary.csv")
    parser.add_argument("--examples", default="results/real_parser_audit_examples.csv")
    parser.add_argument(
        "--examples-cn",
        default="results/real_parser_audit_examples_cn_with_prompts.csv",
    )
    parser.add_argument("--note", default="docs/real_error_analysis_cn.md")
    parser.add_argument("--limit-per-category", type=int, default=10)
    args = parser.parse_args()

    rows = read_jsonl(ROOT / args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    summary = summarize(rows)
    examples = representative_examples(rows, args.limit_per_category)
    write_csv_for_spreadsheets(ROOT / args.summary, summary)
    write_csv_for_spreadsheets(ROOT / args.examples, examples)
    write_titled_csv_for_spreadsheets(
        ROOT / args.examples_cn,
        "真实错误画像人工复核表：Qwen2.5-0.5B base diagnostic",
        chinese_review_examples(examples),
    )
    write_research_note(ROOT / args.note, rows, summary, examples)
    print(
        "wrote "
        f"{args.summary}, {args.examples}, {args.note} "
        f"from {len(rows)} diagnostic rows"
    )


if __name__ == "__main__":
    main()
