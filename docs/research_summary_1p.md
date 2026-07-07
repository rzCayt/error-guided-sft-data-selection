# 一页研究摘要

## 研究问题

小型 base language model 在诊断集上暴露出的错误，能否用于指导 SFT 数据选择？更具体地说，这些错误画像是否能提供超出 matched random sampling 的有效信号？

## 当前实验

我构建了一个可控的数值推理数据集，每条样本都有 solver 生成的确定答案。数据被拆成独立的 `candidate_pool`、`dev_diagnostic` 和 ID/OOD test split。当前只使用 `dev_diagnostic` 做 base model 诊断，不用 test split 调整策略。

第一轮真实诊断使用 `Qwen/Qwen2.5-0.5B`：

- 样本数：100
- 数值准确率：0.21
- 原始输出：`results/real_base_diagnostic_outputs.jsonl`
- 错误画像：`results/real_error_profile.csv`
- 错误分析笔记：`docs/real_error_analysis_cn.md`

按任务族看，`weighted_aggregation` 最弱，25 条全错；`multiplicative_relation` 相对最好，25 条中 11 条正确。

## 初步发现

错误不是单一类型。当前自动审计只是启发式分类，不是人工标注真值；它提示至少有三类信号混在一起：

- 模型确实不会算，或者没有学会某类数值关系。
- 模型输出多个数字或等式，当前 parser 可能取到错误数字。
- 部分题目表达可能诱导模型使用错误公式，例如把 weighted aggregation 当成普通相加。

这提示 error-guided selection 有继续分析的价值，但也说明不能直接把错误画像当成训练数据选择证据。

## 下一步

先做 20-30 条真实错误样例的人工复核，判断哪些错误适合用 SFT 数据修复，哪些应该先处理 prompt/parser。只有当错误画像被确认是稳定的模型弱点，而不只是解析或题目表达问题，才进入 selection bias audit 和 LoRA 对比。

## 当前不能声称

- 不能声称 Targeted selection 已经优于 Random。
- 不能声称 LoRA 已经提升模型。
- 不能把 simulated placeholder 当成真实实验结果。
- 不能把 `dev_diagnostic` 当最终测试集。
