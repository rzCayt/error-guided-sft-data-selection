# 给 Codex 的 Phase 2 v8.2 立即执行指令

你现在接管的是已经完成对抗审计的 Phase 2 v8.2 release。你的首要任务不是改研究问题、优化结果或写申请材料，而是按照冻结协议完成 Q0–Q2，并在人工授权后稳定跑完 24-cell common-mix 矩阵。

## 允许立即开始的工作

1. 在两台独立单卡 4090D/4090 AutoDL 实例部署同一个 release；
2. 从 fresh extract 建立相同 clean Git commit；
3. 安装并核验 pinned Python 3.12 / torch 2.8.0+cu128 环境；
4. 运行完整 Q0；
5. 在两台主机运行 materialized-contract audit、offline dataset-cache qualification 与 static runtime preparation；
6. 完成 GPU0/GPU1 base16、adapter128 推理桥接；
7. 完成 GPU0 A1/A2 与 GPU1 B1 训练锚点；
8. 生成 `READY_FOR_HUMAN_REVIEW.json`；
9. 在该文件生成后停止，提交证据，等待人工口令 `START_PHASE2_V8_COMMON24`。

## 禁止事项

- 禁止改变 24-cell 矩阵；
- 禁止增加或删除 seed、list、method；
- 禁止改选样、token、label、LoRA、学习率、步数或数据预算；
- 禁止 batch>1、vLLM、SGLang、FlashAttention 替换、TF32、DDP、`torch.compile`；
- 禁止只跑 targeted tests 后伪装成 Q0 PASS；
- 禁止在 24/24 完成前查看或汇报方法准确率；
- 禁止根据中间结果停止或调参；
- 禁止放宽 training-anchor 阈值；
- 禁止使用除 `configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json` 外的 canonical authority；
- 禁止写 CV、导师邮件或招聘材料。

## 失败处理

任何 P0/Q0/Q1/Q2 失败：

1. 停止正式矩阵；
2. 保留原始日志、attempt、SHA 和失败状态；
3. 只修复导致失败的工程问题；
4. 不修改科学参数或门槛；
5. 重新生成 versioned release、semantic manifest 和 package manifest；
6. 从受影响的最早闸门重新运行。

## 人工授权后的长期运行

收到精确口令：

```text
START_PHASE2_V8_COMMON24
```

才生成 `RELEASE_GO.json`，然后两台机器各运行一个 worker、12 个 cell。每张卡任何时刻只允许一个 GPU 进程；CPU 打包可与下一格 GPU 运行重叠。SSH 断开时优先检查 heartbeat、PID、state 和 run directory，不得直接重复启动。

24/24 完成后仍不得自动解盲，等待：

```text
PHASE2_V8_CLEAN24_UNBLIND_APPROVED
```

## 成功定义

不是“结果好看”，而是：

- 24/24 cell `COMPLETE`；
- 每格 formal audit、OOD audit、evidence package 均 PASS；
- 0 重复 cell；
- 0 被覆盖 artifact；
- 0 非 canonical 配置；
- 0 中途科学参数变化；
- 完整盲态；
- 可从最终 cell 追溯至 selection、token、label、config、raw output、parser、环境和 SHA。
