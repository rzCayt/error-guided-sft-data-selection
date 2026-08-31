> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# 10,000 条 Tulu 候选的 RDS+ 分批评分说明

## 这一步在做什么

这一步只给冻结的 Tulu 候选样本排序，不训练 LoRA，也不计算新的
GSM8K 准确率。

- `rds_all`：448 条固定诊断题都参与候选排序。
- `rds_error`：只让其中基础模型答错的 99 条题参与候选排序。
- `representation`（表示向量）：把一条文字输入经过固定的
  Qwen2.5-1.5B Base 后，压缩成一个数值向量。两种选择器使用完全相同的
  表示方法，唯一差别是参与排序的查询题集合。
- “可训练候选”：按正式训练的 `max_length=512` 截断后，回答部分仍至少有
  一个需要计算损失的词元。完全只剩提示词、回答被截光的候选不会进入排序。

选择器、查询定义、表示方法、预算 `B=500` 和停止门槛均未改动。

## 为什么分成小块

正式候选接近 10,000 条。程序每次只编码一个不可覆盖的小块：

- 查询每块 64 条；
- 候选每块 128 条；
- 模型推理批量为 1。

每块完成后都会保存：

- 本块候选或查询的固定顺序；
- 表示向量及其 SHA-256；
- GPU 峰值显存；
- 开始、运行中和结束时的温度；
- 模型、数据、代码和配置哈希。

如果进程中断，只需继续计算缺失块；已经完成的块会先验哈希并直接跳过，
不会覆盖。

## 四个稳定入口

以下命令都从仓库根目录运行。

### 1. CPU 来源与可训练性审计

```powershell
python scripts/run_rds_full_pool.py prepare `
  --protocol-config configs/public_gsm8k_v1.json `
  --execution-config configs/rds_full_pool_thermal_v1.json `
  --data-manifest-dir results/research_public_gsm8k_v1/data_manifest_full_v2_fuzzy `
  --query-groups-dir results/research_public_gsm8k_v1/query_groups_diagnostic448_clean_8b5273d `
  --run-dir results/research_public_gsm8k_v1/rds_full_pool_10k_v1
```

### 2. 一次只计算一个小块

```powershell
python scripts/run_rds_full_pool.py encode `
  --run-dir results/research_public_gsm8k_v1/rds_full_pool_10k_v1 `
  --kind candidate `
  --chunk-index 0
```

`--kind` 可取 `query` 或 `candidate`。`--chunk-index` 从 0 开始。

启动前要求：

- GPU 温度不高于 65°C；
- 已占用显存不高于 512 MiB；
- 系统可用内存不少于 6 GiB。

运行中每 8 个批次取一次温度；达到 82°C 会终止当前块且不产生“完成”
清单。该检查只是本次进程的安全保护，不是后台自动监控。

### 3. 查看缺哪些块

```powershell
python scripts/run_rds_full_pool.py status `
  --run-dir results/research_public_gsm8k_v1/rds_full_pool_10k_v1 `
  --deep
```

`--deep` 会同时重新读取表示向量并核验张量哈希。

### 4. 汇总排序

```powershell
python scripts/run_rds_full_pool.py finalize `
  --run-dir results/research_public_gsm8k_v1/rds_full_pool_10k_v1
```

只有查询块和候选块全部完整时才允许汇总。汇总会产生两套完整名次，
不会启动训练。

## 冻结两份 B=500 清单

```powershell
python scripts/freeze_b500_rds_selections.py `
  --run-dir results/research_public_gsm8k_v1/rds_full_pool_10k_v1 `
  --rds-all-output results/research_public_gsm8k_v1/b500_rds_all_selection_v1/selection_manifest.json `
  --rds-error-output results/research_public_gsm8k_v1/b500_rds_error_selection_v1/selection_manifest.json
```

每份清单包含 500 个候选的来源字段、训练总词元、回答监督词元、两种
RDS+ 名次、候选 ID 顺序哈希和完整上游证据哈希。

## 结论边界

这一步最多能证明：

1. 两个冻结选择器在完整候选池上各自选中了哪 500 条；
2. 来源、词元、代码、数据和表示向量可以被审计；
3. 中断后可以从缺失块继续。

它不能证明哪份清单训练后效果更好。只有后续统一配置的正式训练与评估
才能回答该问题；本阶段明确不启动九组正式训练。
