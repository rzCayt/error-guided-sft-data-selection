> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# Phase 1准备状态 v3.2

## 结论

项目没有完成，也没有只剩“跑16格”这一项。当前完成的是正式训练前的CPU合同、OOD数据和盲态控制；正式GPU训练仍为`0/16`。

## 已经完成的新增工作

### 正式矩阵

- OOD绑定版矩阵：`configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json`
- SHA-256：`44d7288f4e785af61f8ebe21ec4ad1883b8b7bd542069c2fae675796724dd29a`
- 16/16选样合同READY；
- 每格仍为500条、约32000唯一回答监督词元、2次曝光、64次优化器更新；
- 没有启动任何正式格子。

### OOD评估冻结

- SVAMP：300题；
- ASDiv numeric-only：2067题；
- MultiArith：155题；
- 合计2522题；
- 216条ASDiv非数值或歧义答案被预先排除；
- 1条与训练/GSM8K参考高度重合的题被排除；
- 46条跨OOD近重复题被确定性去重；
- 三任务分别报告，再计算等权宏平均；
- 原始题面不写入公开manifest。

OOD manifest SHA-256：

`c057be50cfafeb2c24040eec4ad6421d5d66c6c859fc20a960b388f8accce302`

### 盲态控制

- 私有映射只保存在`.aris/control`；
- 公开manifest只显示`method_A/B/C/D`；
- 当前16格均为PENDING；
- 16个正式审计未全部PASS前禁止解盲；
- 16个OOD审计未全部PASS前同样禁止解盲；
- 聚合门槛当前为`BLOCKED_INCOMPLETE`且不输出准确率。

## 仍未完成

### GPU qualification

- 16样本过拟合；
- adapter保存/重载；
- optimizer、scheduler、RNG和checkpoint恢复；
- 128题canary；
- 第二张同型号GPU等价性。

### Phase 1

- 16个正式训练格；
- 每格GSM8K 1319题；
- 每格2522题OOD；
- 独立审计；
- 盲态方差分析；
- 全部完成后统一解盲。

### 论文级后续

- 60格确认矩阵；
- BM25、k-center、Longest和一个外部强selector；
- 候选效用可靠性M0；
- 微集合非加性M1；
- 确认集合M2；
- 8K/128K预算；
- 3B–4B模型复现；
- claim audit与论文。

## 当前停止原因

GPU阶段未开始不是因为任务做完，而是因为用户明确要求：

- Phase 1冻结；
- AutoDL关机；
- 暂不租第二张GPU；
- 不准备CV和教授邮件。

CPU准备可以继续；下一次开机前应先完成云端qualification包和fresh-clone恢复验证。
