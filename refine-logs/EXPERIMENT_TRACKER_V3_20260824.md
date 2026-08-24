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
| V3-P0-010 | 完整CPU回归 | DONE | 265 tests passed；新增相关23 tests passed；ruff PASS |
| V3-P0-011 | 完整相似度artifact | BLOCKED_INPUT | 从旧云端RDS chunks恢复，或单4090重算 |
| V3-P0-012 | 正式近重复簇artifact | BLOCKED_INPUT | 云端读取固定Tulu revision后生成 |
| V3-P0-013 | 四次正式名单与信息闸门 | BLOCKED_BY_011_012 | 16份manifest和gate全部冻结 |
| V3-P0-014 | 双4090 qualification | NOT_STARTED | 两卡等价、重载、恢复和canary通过 |
| V3-P1-001 | Phase 1十六格 | NOT_STARTED | 16/16 `AUDITED_PASS` |

当前不准备CV和教授邮件，也没有启动GPU或付费实例。
