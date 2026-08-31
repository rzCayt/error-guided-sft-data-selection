> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# 云端 qualification 与 OOD 运行手册 v1

## 目的

本手册只负责在正式 Phase 1 前验证云端训练/评估工程链，以及在每个正式 adapter 完成后执行三套算术 OOD 评估。qualification 不是 selector 有效性的证据，也不会自动启动 Phase 1。

## 开机后的固定顺序

### 1. CPU 合约预检

```bash
python scripts/preflight_budget_equivalent_phase1_matrix.py \
  --config configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json

python scripts/run_budget_equivalent_qualification.py --contract-only

for dataset in svamp asdiv_numeric multiarith; do
  python scripts/run_budget_equivalent_ood_eval_worker.py \
    --dataset "$dataset" \
    --contract-only
done
```

必须看到：

- Phase 1 的 16 个 cell 全部 `READY`；
- qualification 为 `READY`，16 条 overfit、128 条 canary；
- SVAMP 300 条、ASDiv numeric 2067 条、MultiArith 155 条；
- 矩阵 SHA-256 为 `44d7288f4e785af61f8ebe21ec4ad1883b8b7bd542069c2fae675796724dd29a`；
- `gpu_accessed=false`。

### 2. 单卡 qualification

```bash
python scripts/run_budget_equivalent_qualification.py
```

统一入口依次执行：

1. 16 条 development 样本的 response-only LoRA 过拟合；
2. adapter 保存和重载；
3. 用真实训练批次验证 adapter、AdamW、scheduler 和随机状态的 checkpoint 恢复；
4. 在固定 128 条 held-out canary 上生成模型回答；
5. CPU 重新解析并生成不可覆盖的 `qualification_audit.json`。

只有最终 `status=PASS` 才允许启动 Phase 1。它仍不说明任何数据选择方法有效。

如果 qualification 中断，保留运行目录，重新执行：

```bash
python scripts/run_budget_equivalent_qualification.py \
  --overfit-run-dir <已有的 qualification run 绝对路径>
```

### 3. 长时间 Phase 1 训练

qualification 通过后，按照冻结 `job_order` 一次只运行一个 cell。训练命令必须显式给出 `cell-id`，不能写自动遍历并在前一格完成后启动下一格。每格依次完成：

1. 64 个 optimizer step 的训练；
2. adapter 保存、重载及哈希；
3. 1319 条 GSM8K；
4. 三套 OOD；
5. CPU 独立审计。

断电或关机后只允许使用原 run directory 恢复，不能创建同一 cell 的第二个运行目录。

### 4. 单个 adapter 的 OOD worker

单卡默认每个任务只开一个 shard：

```bash
for dataset in svamp asdiv_numeric multiarith; do
  python scripts/run_budget_equivalent_ood_eval_worker.py \
    --config configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json \
    --run-dir <正式 cell 的 run 绝对路径> \
    --dataset "$dataset" \
    --shard-index 0 \
    --shard-count 1
done
```

worker 会在加载每一条原始数据后重新核对 source-row、question、answer 和 gold value 的 SHA-256；任何一个哈希不一致都会停止。重复执行同一条命令会从已有合法前缀继续，不覆盖原输出。

完成三套数据后运行：

```bash
python scripts/audit_budget_equivalent_ood.py \
  --config configs/budget_equivalent_phase1_matrix_frozen_20260824_v2.json \
  --run-dir <正式 cell 的 run 绝对路径> \
  --shard-count 1
```

审计只在三套数据都完整时写入 `audit/ood_audit.json`。标准输出不显示准确率；16 个正式 cell 的 GSM8K 和 OOD 审计全部通过后才统一解盲。

## 长时间监督规则

- 每次只运行一个训练或评估进程；
- 每 8 个 optimizer step 保存不可变 checkpoint；
- 训练进程异常退出后先核对 checkpoint、token log 和 adapter hash，再恢复；
- 不因中间准确率修改 selector、学习率、训练步数或 prompt；
- 不自动启动下一格；
- qualification、canary 和 OOD 不消耗正式 selection replicate；
- AutoDL 关机状态下只运行本地 CPU 合约与审计代码。
