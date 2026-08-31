> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# 实验跟踪表

| Run ID | 里程碑 | 目的 | 方法 | Seed | 优先级 | 状态 | 备注 |
|---|---|---|---|---:|---|---|---|
| DEPLOY-001 | M0 | 创建AutoDL 4090实例 | environment | - | MUST | WAIT_USER_CONFIRMATION | 尚未付费或启动 |
| ENV-001 | M0 | 依赖、BF16 witness和两级preflight | environment | 20260722 | MUST | BLOCKED_BY_DEPLOY | spec `4eda29c1` |
| R001 | M1 | 云端主矩阵 | random | 17 | MUST | TODO | 第1项 |
| R002 | M1 | 云端主矩阵 | rds_all | 17 | MUST | TODO | 第2项 |
| R003 | M1 | 云端主矩阵 | rds_error | 17 | MUST | TODO | 第3项 |
| R004 | M2 | 云端主矩阵 | rds_all | 29 | MUST | TODO | 第4项 |
| R005 | M2 | 云端主矩阵 | rds_error | 29 | MUST | TODO | 第5项 |
| R006 | M2 | 云端主矩阵 | random | 29 | MUST | TODO | 第6项 |
| R007 | M3 | 云端主矩阵 | rds_error | 41 | MUST | TODO | 第7项 |
| R008 | M3 | 云端主矩阵 | random | 41 | MUST | TODO | 第8项 |
| R009 | M3 | 云端主矩阵 | rds_all | 41 | MUST | TODO | 第9项 |
| ANALYSIS-001 | M4 | 配对种子与层次bootstrap | all | 17/29/41 | MUST | BLOCKED_BY_R001_R009 | 不启动GPU |
| ERROR-001 | M4 | 共同错误与方法差异案例 | all | 17/29/41 | MUST | BLOCKED_BY_R001_R009 | 不改变parser |
| HW-001 | M4 | 本机与云端对应单元描述性核对 | random/rds_all | 17 | NICE | BLOCKED_BY_R001_R002 | 不作等价性声明 |

当前托管边界：云端准备分支和运行文件在本地完成；GitHub分支尚未推送；付费实例尚未创建。
