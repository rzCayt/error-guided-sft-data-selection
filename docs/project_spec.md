# 项目设计说明

## 标题

基于错误诊断的数据选择：面向小型数值推理语言模型的高效 LoRA SFT

英文标题可保留为：

```text
Error-Guided Data Selection for Data-Efficient LoRA SFT in Small Numerical Reasoning Language Models
```

## 研究问题

在相同样本预算和训练预算下，base model 的诊断错误能否指导 SFT 数据选择，并比 matched random selection 更有效？

## 核心假设

如果 base model 在某些任务族、难度、答案量级、推理长度或错误类型上有稳定弱点，那么从独立候选池中选择更针对这些弱点的样本，可能比只按粗粒度 metadata 随机匹配的样本更有效。

这个假设目前还没有被 LoRA 或 Targeted-vs-Random 对比验证。当前仓库已经完成第一轮真实 `Qwen/Qwen2.5-0.5B` base diagnostic，但它只说明 base model 错误画像；方法有效性结论必须等 parser/error audit、selection bias audit 和后续 LoRA 对比完成后才能讨论。

## 非目标

- 这不是金融大模型项目。
- 这不是金融问答系统。
- 这不是投资建议或交易策略。
- 财务/商业风格文字只是数值推理任务的可控外壳。
- 在真实 LoRA 训练完成并记录前，不声明模型性能提升。

## 数据划分

| Split | 用途 | 是否用于选择 | 是否用于最终评估 |
| --- | --- | ---: | ---: |
| `candidate_pool` | 训练样本候选池 | 是 | 否 |
| `dev_diagnostic` | base model 错误画像 | 是，只使用聚合错误画像 | 否 |
| `test_id` | 锁定的同分布测试 | 否 | 是 |
| `test_ood_template` | 锁定的模板外测试 | 否 | 是 |
| `test_ood_range` | 锁定的数值范围外测试 | 否 | 是 |

选择策略不能读取 test predictions、test metrics，也不能利用 test labels 之外的 split 构造信息来调参。

额外泄漏防护：任何真实训练比较前，都必须运行 near-duplicate audit。当前生成器按 split 使用独立 seed，但模板空间较窄，所以“没有重复 id”不足以证明评估干净。

## 实验流程

1. 生成 solver 可验证的确定性合成样本。
2. 在任何诊断或训练声明前运行 split leakage audit。
3. 在 `dev_diagnostic` 上运行 base diagnostic。
4. 解析模型答案，并映射到 error taxonomy。
5. 按任务族、难度、答案量级、推理长度和错误类型建立错误画像。
6. 从 `candidate_pool` 中选择 targeted subset。
7. 在同样预算下构造 matched-random subset。
8. 条件允许时运行 LoRA smoke/full experiments。
9. 在锁定测试集上比较 Base、Random、Targeted。

## 指标

- 解析成功率。
- 数值 exact accuracy，允许预先定义的 tolerance。
- 必要时报告 MAPE。
- 按任务族和难度分组的 accuracy。
- 错误类型分布。
- 样本预算和 token/训练预算合规性。

## 初始模型选择

从 `Qwen/Qwen2.5-0.5B` 或同级别小型开源 base model 开始。如果本地硬件无法加载模型，保留模型接口并生成 no-training evidence package，不能伪造训练结果。

## 最低真实实验标准

在以下条件全部满足前，不能声明模型提升：

- Base diagnostic 来自真实模型，而不是 `simulate_prediction`。
- Base、Random、Targeted 使用相同 prompt format、parser、numeric tolerance 和 decoding settings。
- B128 至少同时跑 matched-random 与 error-guided；B256 是优先后续预算。
- 选择策略在任何锁定 test evaluation 前冻结。
- `results/main_results_v0.csv` 由真实评估行重新生成，并记录模型名、seed、训练 run id。

## 研究价值

这个项目的价值在于把 error-guided data selection 变成一个可复现实验问题：问题定义、数据生成、诊断测量、数据选择、baseline 控制、泄漏审计、实验记录和克制表述都被显式化。即使最终结果为负，也能说明诊断错误是否不足以提供 metadata matching 之外的有效选择信号。
