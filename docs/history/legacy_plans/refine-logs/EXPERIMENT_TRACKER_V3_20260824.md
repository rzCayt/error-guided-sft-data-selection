> **Historical snapshot.**
>
> This document records an earlier research stage and is not the current result.
>
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# 实验跟踪表 v3

| ID | 内容 | 状态 | 通过标准 |
|---|---|---|---|
| V3-P0-001 | 干净副本与分支 | DONE | 基于`6d5bcea`，旧脏仓库未改 |
| V3-P0-002 | token-weighted loss复核 | DONE | 原实现保留，梯度不依赖micro-batch切分 |
| V3-P0-003 | 恰好64步训练计划 | DONE | 1000次样本曝光分成40×16和24×15 |
| V3-P0-004 | 四方法预算等价求解器 | DONE | 500条、32K±0.5%、common配额及token约束 |
| V3-P0-005 | 真实候选池可行性试算 | DONE | 8542候选、30层、最小自由度9、1.76秒求解 |
| V3-P0-006 | 查询bootstrap | DONE | 保持349/99分层规模并产生独立权重 |
| V3-P0-007 | 相似度导出接口 | DONE | chunk哈希、ID顺序和tensor形状绑定 |
| V3-P0-008 | 近重复聚类接口 | DONE | MinHash-LSH候选发现+真实n-gram复核 |
| V3-P0-009 | Phase 1矩阵和只读审计 | DONE | 16格、64步、adapter、1319输出、哈希 |
| V3-P0-010 | 完整CPU回归 | DONE | 266 tests passed；本次修改文件ruff PASS |
| V3-P0-011 | 完整相似度artifact | DONE | 单4090重算7+67块；448×8542；SHA `994583a61022...58a11` |
| V3-P0-012 | 正式近重复簇artifact | DONE | 8542条、7869簇；SHA `a402b8ad118f...68fd1` |
| V3-P0-013 | 四次正式名单与信息闸门 | DONE | 16份manifest独立审计PASS；核心矩阵允许 |
| V3-P0-014 | 双4090 qualification | NOT_STARTED | 两卡等价、重载、恢复和canary通过 |
| V3-P1-001 | Phase 1十六格 | NOT_STARTED | 16/16 `AUDITED_PASS` |

信息闸门结论：targeted policy gate通过；error-conditioning increment gate失败（最小Top-500更换率7%，最高完整排序Spearman 0.998），因此不运行`rds_all`完整训练矩阵。当前只使用过一张4090D生成Phase 0表示；未启动LoRA训练，不准备CV和教授邮件，第二张GPU按用户要求暂不租用。
