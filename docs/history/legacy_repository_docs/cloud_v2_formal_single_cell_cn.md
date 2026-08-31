> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# Cloud-v2 B=500 正式单格运行手册

## 固定边界

- 训练：单进程，`micro_batch=1`、`gradient_accumulation=16`、有效 batch 16；
- loss：整个有效 batch 的 response-token loss 总和除以 response-token 总数；
- 训练 2 epochs；checkpoint 每 10 个 optimizer steps 及最终 step 63；
- 评估：同一物理 GPU 上两个进程，每个进程始终单题 `generate`，严禁 batch>1；
- test 分片：`[0,660)` 与 `[660,1319)`；
- 任一 worker 失败、GPU UUID 不同、adapter 不同或 1319 条不完整时禁止 PASS 合并；
- 单次入口只接受一个 method 和一个 seed，不会自动运行下一格；
- stdout、registry 和单格 audit 不显示 accuracy 或方法比较。

## 冻结顺序

1. `rds_all / 17`
2. `rds_error / 17`
3. `rds_all / 29`
4. `rds_error / 29`
5. `rds_all / 41`
6. `rds_error / 41`
7. `random / 17`
8. `random / 29`
9. `random / 41`

顺序只由 registry 展示，系统不会自动串行启动。

## CPU dry-run

```bash
python scripts/preflight_cloud_v2_formal_matrix.py
```

必须看到 `automatic_execution=false`、9 个独立命令及上述顺序。

## 运行一格

例如第一格：

```bash
python scripts/run_cloud_v2_formal_cell_fixed.py \
  --method rds_all \
  --seed 17
```

发生中断后只能显式续跑同一目录：

```bash
python scripts/run_cloud_v2_formal_cell_fixed.py \
  --method rds_all \
  --seed 17 \
  --resume-run-dir <原run目录>
```

禁止在同一张卡上同时运行两个训练 cell。两个并发进程只用于该 cell 的正式评估。

## 单格 audit

```bash
python scripts/audit_cloud_v2_formal_cell_fixed.py \
  --run-dir <已完成run目录>
```

Audit 会核验 selected-list SHA、checkpoint 0/63、adapter 保存与重载、两个 worker 的同卡
UUID、1319 条顺序/哈希、parser 重算一致性以及 `next_cell_started=false`。输出只含 PASS/FAIL、
工程检查与哈希，不暴露准确率。

只有当前格 audit PASS 后，才由操作者手工复制 registry 中的下一条命令。九格均 audit PASS
之后，另行使用 sealed analysis 读取 raw outputs；本入口不承担方法比较。
