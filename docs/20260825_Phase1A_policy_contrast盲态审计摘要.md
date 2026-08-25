# 2026-08-25 Phase 1A policy contrast盲态审计摘要

## 目的

在查看任何下游准确率前，检查RDS-error与random是否真的形成不同的训练名单，以及common-mix约束是否把排序自由度压缩到无法识别。

本审计只读取冻结的：

- 8542条可训练候选清单；
- 448条query清单；
- query×candidate相似度；
- 16份selection manifest；
- common-mix设计与配额。

它没有读取GSM8K或OOD的`numeric_correct`，也没有读取方法级准确率。

## Artifact

- 路径：`.aris/compute/budget_equivalent_policy_contrast_v1.json`
- SHA-256：`11695e8aaac9b65f9c72e386abea2ac74f9d0ba534694c1345e062d4a9387f89`
- 状态：PASS
- `accuracy_accessed=false`
- `downstream_results_accessed=false`

## 主要结果

| 比较 | 平均名单Jaccard | 最小名单替换比例 | 平均RDS排名分位提升 | 四个replicate方向均为正 |
|---|---:|---:|---:|---|
| common-mix | 0.0658 | 86.4% | +0.3525 | 是 |
| free-mix | 0.0334 | 92.4% | +0.4520 | 是 |

selection replicate之间的名单稳定性中位数：

| 方法 | 中位pairwise Jaccard |
|---|---:|
| random_common_mix | 0.0678 |
| random_free_mix | 0.0373 |
| rds_error_common_mix | 0.8484 |
| rds_error_free_mix | 0.8709 |

## 当前允许的解释

1. RDS-error和random在free/common两个可行域中都形成了实质不同的名单；
2. common-mix约束没有把两个政策压缩成几乎相同的干预；
3. RDS名单在自身的RDS排名分数上明显高于random名单，说明selector优化目标确实进入了最终名单；
4. RDS的四个query-bootstrap名单高度稳定，而random名单按设计高度不同。

因此，如果16格下游结果没有差异，不能再简单归因为“两个方法实际选了相同样本”或“common约束彻底消除了排序空间”。

## 当前不允许的解释

本审计不能证明：

- RDS分数能够预测真实训练效用；
- RDS一定优于random；
- RDS稳定性意味着下游收益稳定；
- common/free差异构成因果中介分解；
- error-conditioning相对all-query有新增信息。

这些问题仍需16/16解盲结果、P1.5训练方差和M0候选效用可靠性实验回答。

## 证据缺口

冻结candidate inventory只保存最终`supervised_tokens`与`total_tokens`，没有保存能够独立恢复截断率的字段。因此本审计报告token总量，但不事后声称已经验证truncation rate。下一阶段必须在新协议中显式记录截断前后长度和截断标记。
