# Error-Guided SFT Data Selection

> 中文项目名：基于错误诊断的 SFT 数据选择

这个仓库研究一个具体问题：**小型 base model 在诊断集上暴露出的错误，能不能变成 SFT 数据选择信号？**

当前项目还没有声称方法有效。已经完成的是第一轮真实诊断：用 `Qwen/Qwen2.5-0.5B` 跑 `dev_diagnostic`，保存原始输出，并把错误整理成可检查的研究笔记。

<details>
<summary>English short description</summary>

This repository studies whether diagnostic failures from a small base language model can guide SFT data selection. The current evidence is a first real base-model diagnostic with Qwen2.5-0.5B, not a claim that targeted selection improves LoRA SFT.

</details>

## 当前观察

真实 base diagnostic 只在 `dev_diagnostic` 上运行，不使用 test split 调整策略。

- Model: `Qwen/Qwen2.5-0.5B`
- Samples: 100
- Numeric accuracy: 0.21
- Parser: `parse_numeric_last_number_v1`
- Raw outputs: `results/real_base_diagnostic_outputs.jsonl`
- Error profile: `results/real_error_profile.csv`
- 第一轮错误分析：`docs/real_error_analysis_cn.md`

按任务族粗看：

- `multiplicative_relation`: 11/25 correct
- `ratio_change`: 6/25 correct
- `temporal_numeric_constraint`: 4/25 correct
- `weighted_aggregation`: 0/25 correct

这提示错误信号值得继续分析，但不能直接说明 error-guided selection 会优于 random baseline。

## 第一轮错误分析

当前 parser/error audit 是自动启发式分类，不是人工标注真值。它提示错误可能混合了三类问题：

- 模型确实算错或关系理解错误。
- 模型输出多个数字或等式，last-number parser 可能取到中间值或尾部残片。
- 部分 weighted aggregation 题目中，模型像是在相加，而不是做加权平均。

因此下一步不是马上训练 LoRA，而是先人工看 20-30 条真实错误样例，判断哪些错误适合用 SFT 数据修复，哪些应先改 parser 或 prompt。

## 不能下的结论

- 不能说 Targeted selection 已经优于 Random。
- 不能说 LoRA 已经带来提升。
- 不能把 simulated placeholder 表格当成真实模型结果。
- 不能用 `dev_diagnostic` 调整后再把它当最终测试集。

## 快速复现

```powershell
cd E:\RA准备\07_error_guided_sft_repo
python -m pip install -e .[dev]
python scripts/generate_data.py --all
python scripts/audit_splits.py
python scripts/run_real_base_diagnostic.py --model Qwen/Qwen2.5-0.5B --max-new-tokens 32
python scripts/audit_real_diagnostic_errors.py
python scripts/build_selection_sets.py --budget 128 --profile results/real_error_profile.csv --baseline-seeds 20260711,20260712,20260713
pytest -q
```

如果只是检查 pipeline，也可以运行 simulated diagnostic：

```powershell
python scripts/run_base_diagnostic.py
python scripts/build_selection_sets.py --budget 128
python scripts/evaluate_results.py
```

## 主要文件

- `scripts/run_real_base_diagnostic.py`: 运行真实 Qwen base diagnostic。
- `scripts/audit_real_diagnostic_errors.py`: 生成 parser/error audit 和中文研究笔记。
- `docs/real_error_analysis_cn.md`: 第一轮真实错误画像分析。
- `docs/research_summary_1p.md`: 一页研究摘要。
- `docs/contribution_statement.md`: 如何说明个人研究贡献。
- `docs/index.html`: 可切换中英文的 GitHub Pages 项目页。

## 你的参与方式

这个阶段最需要的是研究判断，而不是继续堆功能。建议阅读 `results/real_parser_audit_examples.csv`，逐条判断：

- 这是真正的推理/计算错误吗？
- 这是输出格式或 parser 问题吗？
- 这是题目表达让模型误解了吗？
- 这种错误是否适合用 SFT 数据修复？

这些判断会决定项目下一步是修 parser、调整 prompt，还是进入 selection bias audit 和后续 LoRA 对比。
