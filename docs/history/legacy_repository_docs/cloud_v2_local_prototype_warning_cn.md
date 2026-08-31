> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# Cloud-v2 本地原型警告

## 禁止运行的文件

`scripts/run_b500_cloud_v2_train_calibration.py` 是未验证的本地原型，禁止在
AutoDL、本机或任何正式环境中运行，也不得据此产生吞吐、显存或模型效果结论。

原因：该原型在每个 micro-batch 后执行一次 `nvidia-smi` 采样。`mb1_ga16` 会比
`mb8_ga2` 多执行八倍监控调用，因此原型会人为放大大 micro-batch 的速度优势。

该文件保持为本地未跟踪文件，仅用于记录被否决的实现；后续 `git add`、commit 和
push 必须明确排除它。

## 唯一允许的训练校准入口

使用：

```bash
python scripts/run_b500_cloud_v2_train_calibration_fixed.py \
  --profile mb4_ga4
```

修正版遵守以下规则：

- 固定 64 条 random calibration 数据；
- 四个 profile 都只有 4 个 optimizer updates；
- 温度仅在启动时采样一次，并在每个 optimizer boundary 采样一次；
- 无中断运行时四种 profile 均为 5 次温度采样；
- `compute_seconds_excluding_monitor_and_checkpoint_io` 排除温度查询与 checkpoint I/O；
- `wall_training_loop_seconds` 保留完整训练循环开销；
- 结果只能用于工程校准，不能进入 cloud-v2 正式九格矩阵。

生成批量校准入口为：

```bash
python scripts/run_b500_cloud_v2_generation_calibration.py \
  --training-run-dir <fixed训练校准run目录> \
  --batch-size 8
```

`training-run-dir` 必须来自修正版入口且其 `calibration_metrics.json` 状态为 `PASS`。
