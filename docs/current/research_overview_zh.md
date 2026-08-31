# 当前研究概览

## 问题

在样本数量、回答监督词元和“数据来源 × 回答长度”组成一致时，冻结的 RDS 目标选择策略能否在 LoRA 后训练中优于匹配 Random？

## 已完成证据

- Qwen2.5-1.5B Base，只在回答部分计算损失的 LoRA SFT；
- 2 种方法 × 4 份名单 × 3 个训练种子 = 24 个审计通过的实验格子；
- 每份名单 500 条样本，服从冻结的监督预算和组成约束；
- GSM8K、SVAMP、ASDiv numeric 和 MultiArith 评估；
- Tulu96 与 GSM8K-domain48 候选级测量；
- 冻结的 CPU 回答组成机制审计。

## 结果

GSM8K 上 RDS 相对 Random 为 +0.480 个百分点，95% 区间 [-0.954, +1.889]；三个分布外任务平均值为 -0.094 个百分点，区间 [-1.316, +1.149]。现有证据既不足以支持有效或有害，也不足以支持等效。

## 下一步

State Dependence v3 先在同一个固定 zero-LoRA 状态下重复测量候选效用。只有固定状态可靠性通过门槛，才进入跨状态比较。CPU 合同和预检已经完成，目前没有 v3 GPU 结果。

详细内容见[结论—证据账本](claim_evidence_ledger.md)、[研究时间线](../research_timeline.md)和[规范结果 JSON](../../results/public_summary/main_results.json)。
