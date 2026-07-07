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

- `parser_or_output_format_risk`: 26 条。模型输出了多个数字或等式，当前 `parse_numeric_last_number_v1` 可能把中间值或尾部残片当作答案。
- `model_calculation_or_reasoning_error`: 30 条。模型给出单一数值但与 solver label 不一致，更像直接算错或关系理解错误。
- `prompt_or_task_misunderstanding_risk`: 23 条。主要集中在 weighted aggregation，模型常把两个数相加，而不是按权重加权。

## 代表样例

- `dev_diagnostic-0000` (ratio_change, easy): answer=30.6, parsed=36.0, category=parser_or_output_format_risk; output:  36 - 15% of 36 | = 36 - 0.15 * 36 | = 36
- `dev_diagnostic-0002` (weighted_aggregation, easy): answer=72.87, parsed=150.0, category=prompt_or_task_misunderstanding_risk; output:  150.00000000000001
- `dev_diagnostic-0003` (temporal_numeric_constraint, easy): answer=93.0, parsed=106.0, category=model_calculation_or_reasoning_error; output:  106
- `dev_diagnostic-0004` (ratio_change, medium): answer=203.0, parsed=193.0, category=parser_or_output_format_risk; output:  145 + 145 * 0.40 = 193.
- `dev_diagnostic-0006` (weighted_aggregation, medium): answer=43.326, parsed=100.0, category=prompt_or_task_misunderstanding_risk; output:  100.000
- `dev_diagnostic-0008` (ratio_change, hard): answer=200.2, parsed=190.0, category=parser_or_output_format_risk; output:  154 + 154 * 0.30 = 154 + 46.2 = 190.
- `dev_diagnostic-0009` (multiplicative_relation, hard): answer=15120.0, parsed=1260.0, category=model_calculation_or_reasoning_error; output:  1260
- `dev_diagnostic-0010` (weighted_aggregation, hard): answer=49.084, parsed=100.00000000000001, category=prompt_or_task_misunderstanding_risk; output:  100.00000000000001
- `dev_diagnostic-0011` (temporal_numeric_constraint, hard): answer=180.0, parsed=181.0, category=parser_or_output_format_risk; output:  138 + 36 + 1 - 29 + 43 = 181
- `dev_diagnostic-0012` (ratio_change, easy): answer=167.9, parsed=1.0, category=parser_or_output_format_risk; output:  146 + 15% of 146 = 146 + 1.5 * 146 = 1
- `dev_diagnostic-0014` (weighted_aggregation, easy): answer=49.088, parsed=62.00000000000001, category=prompt_or_task_misunderstanding_risk; output:  62.00000000000001
- `dev_diagnostic-0015` (temporal_numeric_constraint, easy): answer=110.0, parsed=100.0, category=parser_or_output_format_risk; output:  68 + 28 + 14 = 100

## 对研究问题的影响

这轮结果提示 error-guided selection 值得继续分析，但不能直接进入方法有效性声明。真实错误画像里至少混合了三种信号：模型确实不会算、输出格式导致 parser 可能误判、以及题目措辞可能诱导模型用错公式。下一步应先让人工看 20-30 条错误样例，确认哪些错误适合用 SFT 数据修复，哪些应先改 prompt/parser。

## 你的参与方式

你在这个阶段最适合做研究负责人，而不是数据提供者。建议你直接阅读 `results/real_parser_audit_examples.csv`，给每类错误写一句中文判断：这是真推理错误、格式/parser 风险，还是题目理解问题。你的判断会决定下一步是改 parser、微调 prompt，还是进入 selection bias audit。
