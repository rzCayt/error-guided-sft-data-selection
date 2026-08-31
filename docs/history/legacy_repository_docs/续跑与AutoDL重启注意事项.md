> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# 续跑与 AutoDL 重启注意事项

更新日期：2026-08-25

## 1. 关机前必须做什么

1. 向当前 `screen` 中的正式 worker 发送 `Ctrl-C`，不要直接强制释放实例。
2. 确认：
   - `screen -ls` 不再显示该 worker；
   - `nvidia-smi` 显存接近 0、GPU utilization 为 0；
   - `raw_outputs.jsonl` 行数能够正常读取；
   - 未完成任务不应存在 `metrics.json`。
3. 记录暂停行数、cell ID、dataset、shard index/count、run directory 和 Git commit。
4. AutoDL 控制台只点“关机”，不要点“释放实例”或删除数据盘。

worker 每生成一条就写入并 `fsync`。恢复时会验证已有输出是冻结数据的严格前缀，然后从下一条继续。

## 2. 本次遇到的问题

第三格 ASDiv 在 989/2067 时正常暂停。AutoDL 重新开机后，实例分配到的新物理 GPU UUID 从：

```text
GPU-379ca125-d24d-f627-e184-4b47c1894ee7
```

变为：

```text
GPU-ffed2242-5e3e-2964-d469-966b41be0917
```

旧版 `budget-equivalent-ood-eval-worker-v1` 把 GPU UUID 写入不可变 worker manifest，并要求恢复时整份 manifest 完全一致。因此首次恢复在写入任何新结果前 fail closed：

```text
ValueError: OOD worker manifest changed
```

原始 989 条没有损坏或重算。

## 3. 正确修复

提交 `f0d5ba21d46229ae1a8b2ce232a0d63799da56ad` 增加以下规则：

- manifest 完全相同：正常恢复；
- 只有 GPU UUID 不同：允许恢复；
- adapter、cell、run、matrix、OOD manifest、records、dataset、shard、batch 或设备索引任一不同：继续拒绝；
- 恢复前必须重新验证现有 raw output 前缀；
- 记录 `previous_gpu_uuid`、`current_gpu_uuid`、`validated_prefix_count`、恢复代码 commit 和时间。

本次恢复证据：

```text
validated_prefix_count = 989
resume_worker_code_commit = f0d5ba21d46229ae1a8b2ce232a0d63799da56ad
```

完整 CPU 测试：297 passed；远端定向测试：6 passed。

## 4. 当前已知的小格式问题

提交 `f0d5ba2` 调用 `_append_jsonl` 时传入了单元素 list，因此 `runtime_attempts.jsonl` 当前一行是 JSON 数组，而不是单个 JSON object。

这不影响训练、生成、断点、raw output 或正式指标，但会使恢复事件解析接口不够整洁。处理规则：

- 不在当前第三格运行中修改远端代码；
- 第三格双审计完成后，把参数改成单个 event dict；
- 补测试确保每行都是 JSON object；
- 新提交只影响 provenance 记录格式，不修改科研计算。

## 5. 标准恢复检查表

恢复前必须逐项确认：

- [ ] Git HEAD 与预期一致，worktree clean；
- [ ] formal audit 已通过（若主任务已完成）；
- [ ] 已完成 OOD 数据集的行数与 metrics 均完整；
- [ ] 未完成数据集有 raw prefix、没有 final metrics；
- [ ] selection、adapter、matrix、dataset records SHA 未变化；
- [ ] shard index/count 与原任务一致；
- [ ] 若 GPU UUID 变化，使用 `f0d5ba2` 或更新版本，并保存 runtime attempt；
- [ ] 启动后行数必须从原暂停值继续增长，而不是回到 0；
- [ ] GPU 温度、显存、利用率正常；
- [ ] 完成后仍需 formal + OOD 双审计，不能因恢复成功直接标记 `AUDITED_PASS`。

## 6. 并行与资源利用规则

- 同一 cell 可在已完成逐条等价校准后，并行运行不同冻结 OOD 数据集，以填充闲置 GPU；
- 不对已运行到一半的 shard 临时更改 `shard_count`；
- 不修改 batch、prompt、parser、tokenizer、precision 或生成参数；
- CPU 用于 parser 重算、SHA、归档、下一 cell preflight；
- 内存主要作为 page cache，始终保留安全余量；
- 判断标准是每个 `FULLY_AUDITED_PASS` cell 的时间、成本和失败率，不是表面硬件占用率。
