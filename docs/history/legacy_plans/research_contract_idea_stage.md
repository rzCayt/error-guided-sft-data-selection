> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# Research Contract: Budget-Equivalent Targeted Instruction Selection

**冻结日期：** 2026-08-24
**基础提交：** `6d5bcea76af0d46bfd773f93e8eda072a63e2596`
**开发分支：** `research/budget-equivalent-v3`

## 核心问题

在500条样本、约32,000个唯一回答监督词元、约64,000个训练曝光词元和64次优化器更新一致时：

1. `rds_error_common_mix` 是否优于 `random_common_mix`；
2. 自由数据组成下的总策略效应，是否不同于控制来源和回答长度后的排序效应；
3. 候选级utility为什么没有转化为集合级训练收益。

## 主张边界

- Phase 1十六格只提供工程验收、描述性方向和名单方差，不提供最终有效、等效或无效结论。
- 错误条件化只有通过独立信息性闸门后，才能被描述为相对all-query的新增政策。
- GSM8K为主任务指标；SVAMP、ASDiv、MultiArith约束泛化表述。
- 所有负结果保留；不依据准确率修改selector、配额、parser或停止门槛。

## 预注册门槛

- 监督词元误差不超过0.5%；
- common-mix的prompt和非padding总词元各自距离冻结目标不超过1%；
- 每个来源×长度层候选量/配额不低于4；
- 被迫入选比例不超过10%；
- 近重复簇每簇最多一条；正式运行禁止exact-prompt fallback；
- 实际意义门槛固定为1个百分点。

## AI贡献边界

Codex负责工程脚手架、测试、协议检查和运行协助。研究问题、资源授权、最终结论接受、教授沟通和闭卷技术说明由曹锐哲本人承担。未经人工核验的代码或结果不能表述为本人独立实现。
