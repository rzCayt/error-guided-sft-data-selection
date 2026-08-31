> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# 实验计划 v3：预算等价与集合效用

## Claim Map

| 主张 | 最低可信证据 | 当前状态 |
|---|---|---|
| C1：共同数据组成下RDS排序具有增量 | 8个独立名单×2训练seed×GSM8K/OOD | 尚未运行 |
| C2：总策略效应和分层内排序效应不同 | free/common两组均完成并审计 | 尚未运行 |
| C3：候选utility到集合utility存在可解释非加性 | 候选分组交叉验证和确认集合 | 后续阶段 |

## Phase 0：必须先完成

1. 从旧完整RDS运行目录导出448×8542相似度矩阵；
2. 从固定Tulu revision生成8542条近重复簇清单；
3. 冻结两个输入SHA-256；
4. 生成4个selection replicate下的16份名单；
5. 运行targeted-policy和error-vs-all信息性闸门；
6. 完成双4090 GPU qualification；
7. 冻结16格矩阵配置。

## Phase 1：教授联系前十六格

```text
4 methods × 4 selection replicates × train seed 17 = 16 cells
```

中间准确率保持盲态。每格顺序为训练、adapter保存、重载、1319题评估、CPU审计。16/16全部`AUDITED_PASS`后才允许统一解盲。

## 正式运行入口

```powershell
python scripts/preflight_budget_equivalent_v3.py --config <frozen-protocol>
python scripts/build_budget_equivalent_lists.py --config <frozen-protocol>
python scripts/freeze_budget_equivalent_phase1_matrix.py `
  --protocol-config <frozen-protocol> `
  --selection-root <selection-root> `
  --output-config <new-matrix-config>
python scripts/run_budget_equivalent_cell_v3.py `
  --config <matrix-config> --cell-id <cell-id>
python scripts/audit_budget_equivalent_cell.py `
  --config <matrix-config> --cell-id <cell-id> --run-dir <run-dir>
```

不得直接调用内部实现文件`run_budget_equivalent_cell.py`；公开入口固定为`run_budget_equivalent_cell_v3.py`。

## 当前计算边界

- 本机GPU训练继续冻结；
- 正式Phase 1前只需一张4090恢复/重算表示；
- 16格开始后再使用两张同型号4090独立并行；
- 不使用DDP，不使用PRO 6000；
- error-conditioning闸门失败时不运行rds_all完整训练。
