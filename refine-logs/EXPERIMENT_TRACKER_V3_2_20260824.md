# 实验跟踪表 v3.2

| ID | 阶段 | 内容 | 状态 | 通过标准 |
|---|---|---|---|---|
| V32-R0-001 | CPU | Phase 0 artifacts本地恢复 | DONE | 三个核心SHA与云端一致 |
| V32-R0-002 | CPU | 16格matrix config冻结 | DONE | config SHA=`d6d97e...ac362` |
| V32-R0-003 | CPU | 16格contract审计 | DONE | 16/16 READY；gpu_accessed=false |
| V32-R0-004 | CPU | 私有盲态映射 | DONE | 公开manifest不含真实方法名 |
| V32-R0-005 | CPU | 匿名registry | DONE | 0/16 AUDITED_PASS；禁止解盲 |
| V32-R0-006 | CPU | qualification合同 | DONE | 单卡和第二卡门槛冻结 |
| V32-R0-007 | CPU | OOD revisions与数据manifest | TODO | SVAMP/ASDiv/MultiArith固定ID、SHA、license |
| V32-R0-008 | CPU | OOD prompt/parser测试 | TODO | 同一生成与解析政策；malformed测试通过 |
| V32-R0-009 | CPU | 盲态结果聚合器 | TODO | 只读AUDITED_PASS；16格前不输出准确率 |
| V32-R0-010 | CPU | 云端qualification包 | TODO | fresh clone命令、hash和恢复说明完整 |
| V32-R0-011 | 审计 | 外部独立实验审计 | BLOCKED_TOOL | 当前无可用外部reviewer backend；不得伪称已完成 |
| V32-R1-001 | GPU | 单4090 16样本过拟合 | FROZEN | 用户重新授权开机后运行 |
| V32-R1-002 | GPU | adapter保存/重载 | FROZEN | loss差≤1e-6 |
| V32-R1-003 | GPU | checkpoint/RNG恢复 | FROZEN | 中断恢复一致性通过 |
| V32-R1-004 | GPU | 128题canary | FROZEN | 128输出、parser重算一致 |
| V32-R1-005 | GPU | 第二张同型号卡等价性 | FROZEN | 用户重新授权租卡后运行 |
| V32-R2-001 | Phase 1 | 16格正式矩阵 | FROZEN | 16/16 AUDITED_PASS |
| V32-R2-002 | Phase 1 | 盲态方差审计 | BLOCKED_BY_R2_001 | 方法标签仍匿名 |
| V32-R2-003 | Phase 1 | 统一解盲 | BLOCKED_BY_R2_002 | 16格完整且审计通过 |
| V32-R3-001 | Phase 2 | 60格确认矩阵 | NOT_STARTED | common/free/诊断基线完整 |
| V32-R3-002 | Phase 2 | OOD与扰动评估 | NOT_STARTED | 任务级与macro结果完整 |
| V32-R4-001 | 机制 | M0候选效用可靠性 | NOT_STARTED | 96×3，ICC门槛 |
| V32-R4-002 | 机制 | M1微集合非加性 | NOT_STARTED | 约120集合 |
| V32-R4-003 | 机制 | M2确认集合 | NOT_STARTED | 48×3，grouped CV |
| V32-R5-001 | 泛化 | 第二/第三监督预算 | NOT_STARTED | 8K/128K筛查与门控扩展 |
| V32-R5-002 | 泛化 | 3B–4B复现 | NOT_STARTED | 1.5B机制冻结后16格 |
| V32-R6-001 | 论文 | 投稿级结果与claim audit | NOT_STARTED | 全证据链完成 |

当前正式GPU训练完成度仍是`0/16`，不是因为项目结束，而是因为用户明确冻结Phase 1并关闭实例。
