# B=500 正式矩阵：单任务运行与恢复

本说明只适用于一次手动启动一个任务。当前首个任务固定为：

- 方法：`random`
- 训练随机种子：`17`
- 选择预算：`500`
- 其余八组不会被脚本自动启动

## 1. 启动前检查

先确认九组矩阵的配置、数据、选择清单和文件哈希仍然一致：

```powershell
python scripts/preflight_b500_formal_matrix.py
```

再对首个任务执行只读资源检查。该命令不创建正式运行目录，也不加载模型：

```powershell
python scripts/run_b500_formal_resumable.py `
  --matrix-config configs/b500_formal_matrix_v1.json `
  --strategy random `
  --seed 17 `
  --preflight-only
```

只有输出 `status: READY` 才能启动。

## 2. 正式启动

```powershell
python scripts/run_b500_formal_resumable.py `
  --matrix-config configs/b500_formal_matrix_v1.json `
  --strategy random `
  --seed 17
```

脚本使用全局进程锁，同一时间只允许一个正式 B=500 任务运行。相同的
`random / seed=17` 已存在时，脚本拒绝创建第二份结果，必须显式恢复原任务。

## 3. 温控与恢复

- 启动模型前温度必须不高于 65°C。
- 每个训练 micro-batch 和每道 GSM8K 题目前后都检查温度。
- 75°C 开始暂停计算，冷却到 62°C 后继续。
- 80°C 立即退出本次进程。
- 每次优化器更新后保存一份新的不可覆盖 checkpoint。
- GSM8K 每完成一题就落盘并刷新，因此评估恢复时只继续未完成的后缀。

如果进程以状态 `THERMAL_STOP` 结束，复制输出中的绝对运行目录并执行：

```powershell
python scripts/run_b500_formal_resumable.py `
  --matrix-config configs/b500_formal_matrix_v1.json `
  --strategy random `
  --seed 17 `
  --resume-run-dir "<输出中的运行目录>"
```

恢复过程必须使用同一个 Git 提交、同一矩阵、同一清单和同一执行配置。任一哈希
变化都会拒绝恢复。

## 4. 完成标志

只有运行目录中同时出现以下文件，才算工程闭环完成：

- `training_complete/training_metrics.json`
- `training_complete/adapter/adapter_model.safetensors`
- `evaluation/raw_outputs.jsonl`，严格为 1,319 行
- `evaluation/metrics.json`
- `run_complete.json`

`run_complete.json` 只说明首个任务完整结束，不允许据此比较三个选择器，也不允许
自动启动下一组。
