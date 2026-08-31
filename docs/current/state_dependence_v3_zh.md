# Candidate-Utility State Dependence v3：研究与执行说明

日期：2026-08-31

## 一句话研究问题

候选样本在标准化 zero-LoRA 状态下的一步局部效用排序，能否迁移到真实完成 LoRA 后训练的 adapter 状态？

这里的“效用”定义为：用一个候选样本做一次相同的局部更新后，独立 128 题 utility set 的 gold-response loss 降低量。

## 为什么升级 v2

v2 计划合并历史 seed17/29 与新 runner 测量，但两者的随机性定义不同：历史流程让 seed 同时改变 LoRA 初始化和 probe 随机路径；新 runner 让多个 probe seed 共用同一 adapter 快照。直接合并会把不同 estimand 当成重复测量。

v3 做四项修复：

1. U0 主统计不复用历史 utility，48×3=144 条全部重新测量；
2. U0a 固定同一个 zero-LoRA 初始化，只改变 probe seed，用于测量固定状态下的重复性；
3. LoRA 初始化敏感性单列为可选 U0b，不与 U0a 混合；
4. U1 面板排除四个目标 adapter 训练名单的并集，并在两个 matched probe seeds 上复现。

## 已完成的 CPU 证据

- 恢复并 SHA 核验四个初始 adapter 的 selection manifests；
- 四名单训练样本并集：1,375 条；
- 96 候选中曾被至少一个目标 adapter 训练过：14 条；
- 仍未见候选：82 条；
- 每个 error-score 四分位未见候选数：18、18、22、24；
- 新面板确定性选择 48 条，每四分位 12 条；
- 新面板与四个训练名单重合：0；
- panel selected-ID SHA-256：`eb8440744cb73ed0582becc6559463bf136fd308823fe95eb286d6adebd3bc23`；
- CPU preflight：`READY_FOR_GPU_QUALIFICATION`；
- 两候选×三seed合同：6 条全新测量、0 条历史复用；
- 正式 U0a 合同：144 条全新测量、0 条历史复用；
- 单个 U1 状态合同：96 条全新测量、0 条历史复用。

## 冻结实验顺序

### Q0：GPU qualification

```text
2 candidates × probe seeds {17,29,41}
= 6 measurements
```

检查 snapshot SHA、restore loss、fresh run/resume、显存和输出一致性。qualification 不进入正式统计。

### U0a：固定状态测量可靠性

```text
48 candidates × probe seeds {17,29,41}
= 144 new measurements
```

标准 zero-LoRA 初始化 seed 固定为 17。主要报告 ICC(A,1)、三对 Spearman、bootstrap 区间和候选内 SD。

只有 U0a 达到冻结门槛才进入 U1。

### U1：zero-to-final 跨状态迁移

```text
4 final adapter states
× 48 universal-unseen candidates
× probe seeds {17,41}
= 384 new measurements
```

初始四状态：rep1/rep4 的 Random/RDS、train seed17。每状态独立运行、独立审计；单 seed 只能称为 screen，不能形成 state-dependence/stability 结论。

## 决策分支

- U0a 不可靠：停止跨状态主张，研究 utility measurement uncertainty；
- U0a 可靠、U1 state-dependent：加入中间 checkpoint，研究 revaluation/abstention；
- U0a 可靠、U1 stable：停止 state-dependence，转向 micro-set 非加性；
- U1 模糊：只运行冻结的 rep2/rep3 扩展，不改候选和门槛。

## 结论边界

当前没有 U0/U1 GPU 结果，因此不能声称 state dependence 已存在。即使 U1 成立，也只能说明固定参数状态附近的一步局部效用迁移性有限；不能声称测量了历史 optimizer trajectory，也不能据此解释全部 H1a→H1b 断裂。

## 算力

U0 使用一张 4090/4090D。U1 单卡可完成；若需要缩短墙钟，使用两张同型号卡按 adapter state 独立分片，不使用 DDP。
