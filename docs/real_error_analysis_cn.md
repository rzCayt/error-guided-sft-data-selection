# 真实错误画像第一轮研究笔记

## 当前观察

本轮只分析 `Qwen/Qwen2.5-0.5B` 在 `dev_diagnostic` 上的真实 base diagnostic。样本数为 100，正确 21，错误 79，数值准确率为 0.21。这不是 LoRA 结果，也不是 Targeted selection 优于 Random 的证据。

按任务族粗看：

- `multiplicative_relation`: 11/25 correct, accuracy=0.44
- `ratio_change`: 6/25 correct, accuracy=0.24
- `temporal_numeric_constraint`: 4/25 correct, accuracy=0.16
- `weighted_aggregation`: 0/25 correct, accuracy=0.00

## 错误类型的初步判断

下面分类是自动启发式审计，不是人工标注真值。它的作用是帮助挑选需要人工复核的样例，而不是直接定义最终错误类型。

- `parser_tail_fragment_risk`（parser 尾部残片风险）: 1 条。
- `percent_change_calculation_error`（百分比变化计算错误）: 18 条。
- `weighted_formula_error`（加权公式/聚合方式错误）: 25 条。
- `temporal_calculation_error`（时间顺序加减计算错误）: 21 条。
- `multiplication_calculation_error`（乘法计算错误）: 14 条。

## 代表样例

- `dev_diagnostic-0000` (ratio_change, easy): answer=30.6, parsed=36.0, category=percent_change_calculation_error; output:  36 - 15% of 36 | = 36 - 0.15 * 36 | = 36
- `dev_diagnostic-0002` (weighted_aggregation, easy): answer=72.87, parsed=150.0, category=weighted_formula_error; output:  150.00000000000001
- `dev_diagnostic-0003` (temporal_numeric_constraint, easy): answer=93.0, parsed=106.0, category=temporal_calculation_error; output:  106
- `dev_diagnostic-0004` (ratio_change, medium): answer=203.0, parsed=193.0, category=percent_change_calculation_error; output:  145 + 145 * 0.40 = 193.
- `dev_diagnostic-0006` (weighted_aggregation, medium): answer=43.326, parsed=100.0, category=weighted_formula_error; output:  100.000
- `dev_diagnostic-0008` (ratio_change, hard): answer=200.2, parsed=190.0, category=percent_change_calculation_error; output:  154 + 154 * 0.30 = 154 + 46.2 = 190.
- `dev_diagnostic-0009` (multiplicative_relation, hard): answer=15120.0, parsed=1260.0, category=multiplication_calculation_error; output:  1260
- `dev_diagnostic-0010` (weighted_aggregation, hard): answer=49.084, parsed=100.00000000000001, category=weighted_formula_error; output:  100.00000000000001
- `dev_diagnostic-0011` (temporal_numeric_constraint, hard): answer=180.0, parsed=181.0, category=temporal_calculation_error; output:  138 + 36 + 1 - 29 + 43 = 181
- `dev_diagnostic-0012` (ratio_change, easy): answer=167.9, parsed=1.0, category=percent_change_calculation_error; output:  146 + 15% of 146 = 146 + 1.5 * 146 = 1
- `dev_diagnostic-0014` (weighted_aggregation, easy): answer=49.088, parsed=62.00000000000001, category=weighted_formula_error; output:  62.00000000000001
- `dev_diagnostic-0015` (temporal_numeric_constraint, easy): answer=110.0, parsed=100.0, category=temporal_calculation_error; output:  68 + 28 + 14 = 100

## 对研究问题的影响

这轮结果提示 error-guided selection 值得继续分析，但不能直接进入方法有效性声明。真实错误画像里至少混合了三种信号：模型确实不会算、输出格式导致 parser 可能误判、以及题目措辞可能诱导模型用错公式。下一步应先让人工看 20-30 条错误样例，确认哪些错误适合用 SFT 数据修复，哪些应先改 prompt/parser。

## 你的参与方式

你在这个阶段最适合做研究负责人，而不是数据提供者。建议你直接打开 `results/real_parser_audit_examples_cn_with_prompts.csv`，按中文表头逐条判断：这是真推理错误、格式/parser 风险，还是题目理解问题。你的判断会决定下一步是改 parser、微调 prompt，还是进入 selection bias audit。
