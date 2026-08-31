# Phase 2 v8 完整 Release 使用说明

本 release 只支持两台独立的单卡 RTX 4090D 实例；每台运行一个 worker、12 个 cell。它不支持单卡模式、同机双卡共享资源、DDP、batch>1、vLLM、FlashAttention、TF32 或运行中修改代码。

`configs/`、`results/` 和 `workflow/` 中保留了全仓测试所需的历史夹具。它们是 test-only/noncanonical 资产；正式运行只能读取 `CANONICAL_RUNTIME_FILES_v8_RELEASE.json` 列出的角色和 SHA，不能用历史配置替代。

## CPU Q0

必须把 release 内容部署到已提交且干净的 Git checkout，并保留原始 `phase2_v8_autodl_deployment.tar.gz`。模型和 pinned dataset cache 应先预置在 repo 外；Q0 本身强制离线。唯一入口：

```bash
bash scripts/phase2_v8_cpu_release_gate.sh \
  "$REPO_ROOT" \
  "$REPO_ROOT/DEPLOYMENT_MANIFEST.json" \
  "$RELEASE_ARCHIVE" \
  "$Q0_OUTPUT"
```

Q0 会重新计算 deployment、canonical 和当前冻结语义源码清单的 SHA，运行全量 pytest、v8 定向测试、ruff、failure injection，以及24格 contract-only。旧的 PASS 报告不能替代本命令。GPU environment 还会逐项核验 requirements 中与 tokenization、数据读取和 adapter 序列化相关的完整版本集合。

## 数据集预置与离线资格

fresh host 没有 cache 时，可在正式资格前用 `stage_phase2_v8_offline_datasets.py` 按 pinned revision 预置；随后必须设置 `HF_DATASETS_OFFLINE=1`，由 `qualify_phase2_v8_offline_datasets.py` 对四个任务逐条核验。网络预置报告不能替代离线 qualification。

## GPU Q1/Q2

Q0 PASS 后才运行：

1. 两机离线数据集 cache qualification 与 environment manifest；
2. base16、archived-adapter128 推理桥接；
3. GPU0 A1；
4. GPU0 fresh-process A2；
5. GPU1 B1；
6. A1/A2 同卡比较；
7. A1/B1 跨卡比较；
8. 三个 anchor 的128题逐 token 比较。

Q2 结束后只生成 `READY_FOR_HUMAN_REVIEW.json`，不得自动开始正式矩阵。四个正式评估数据集必须先在 `HF_DATASETS_OFFLINE=1` 下逐条核验，dataset-cache contract SHA 会写入两台机器的 environment contract。

## 人工放行

用户明确发送 `START_PHASE2_V8_COMMON24` 后，运行：

```bash
python scripts/authorize_phase2_v8_release.py \
  --ready READY_FOR_HUMAN_REVIEW.json \
  --operator-confirmation START_PHASE2_V8_COMMON24 \
  --output RELEASE_GO.json
```

正式 worker 和每个 cell 都会重新验证 RELEASE_GO、原始压缩包、deployment tree、semantic code tree、Git commit、模型树、tokenizer树、environment、backend 和训练锚点。任何 SHA 不一致都会停止。

## 正式长跑

获得 `RELEASE_GO.json` 后，两台实例分别运行 `scripts/phase2_v8_run_worker.sh`。准确率保持盲态，24/24 完成前不能统一解盲。
