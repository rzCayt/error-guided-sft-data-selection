# Phase 2 v8.2：AutoDL 启动顺序

本文件用于交给 Codex/实验执行代理。不得跳步，不得解盲中间准确率。

## 0. 不可变研究设计

```text
2 methods × 4 list replicates × 3 train seeds = 24 cells
methods = random_common_mix, rds_error_common_mix
seeds = 17, 29, 41
GPU = two independent single-GPU RTX 4090D/4090 hosts
batch = 1, natural per-example padding
```

禁止修改矩阵、选样、token、label、LoRA、学习率、optimizer steps、prompt、parser、generation、OOD 数据集或统计门槛。

## 1. fresh extract 与 clean Git

在两台机器上从同一个 deployment archive 解压到空目录。初始化同一 Git commit，确认：

```bash
git status --porcelain
# 必须为空
```

两台机器的 commit SHA 必须相同。

## 2. 预置模型与冻结数据集 cache

在进入任何正式 gate 前，先把指定 model revision 和 pinned datasets 预置到本地 cache。若 fresh host 尚无数据集 cache，可暂时允许网络：

```bash
unset HF_DATASETS_OFFLINE
python scripts/stage_phase2_v8_offline_datasets.py \
  --config configs/phase2_clean_common24_v8_canonical.json \
  --output "$HOST_PREFLIGHT_ROOT/dataset_cache_staging.json"
```

同时预下载并固定 `Qwen/Qwen2.5-1.5B` revision：

```text
8faed761d45a263340a0528343f099c05c9a4323
```

预置完成后立即关闭网络依赖：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

`STAGED` 不等于资格通过；后续离线 qualification 会逐条验证。cache 与 model snapshot 应放在 repo 外，不得污染 clean Git tree。

## 3. 运行 Q0

在 pinned Python 3.12 / torch 2.8.0+cu128 环境执行：

```bash
bash scripts/phase2_v8_cpu_release_gate.sh \
  "$REPO_ROOT" \
  "$REPO_ROOT/DEPLOYMENT_MANIFEST.json" \
  "$RELEASE_ARCHIVE" \
  "$Q0_OUTPUT"
```

Q0 强制 offline。任一检查 FAIL，停止。不得仅运行 targeted tests 后宣称 Q0 PASS。

## 4. 每台机器执行 host preparation

```bash
bash scripts/phase2_v8_prepare_host.sh \
  "$REPO_ROOT" \
  "$MODEL_SNAPSHOT" \
  "$MATERIALIZED_CONTRACTS" \
  "$STATIC_ROOT" \
  "$HOST_PREFLIGHT_ROOT" \
  "$SEMANTIC_MANIFEST"
```

必须生成并通过：

```text
materialized_contract_audit.json
dataset_cache_qualification.json
static_runtime.json
```

## 5. GPU0 / GPU1 推理资格

GPU0：

```bash
bash scripts/phase2_v8_qualify_gpu0.sh \
  "$REPO_ROOT" "$MODEL_SNAPSHOT" "$ARCHIVED_ADAPTER" \
  "$STATIC_ROOT" "$SESSION_ROOT" gpu0 "$GPU0_UUID" \
  "$SEMANTIC_MANIFEST" \
  "$HOST_PREFLIGHT_ROOT/dataset_cache_qualification.json"
```

GPU1：

```bash
bash scripts/phase2_v8_qualify_gpu1.sh \
  "$REPO_ROOT" "$MODEL_SNAPSHOT" "$ARCHIVED_ADAPTER" \
  "$STATIC_ROOT" "$SESSION_ROOT" gpu1 "$GPU1_UUID" \
  "$GPU0_BASE_SIGNATURES" "$GPU0_ADAPTER_SIGNATURES" \
  "$SEMANTIC_MANIFEST" \
  "$HOST_PREFLIGHT_ROOT/dataset_cache_qualification.json"
```

随后运行 `finalize_phase2_v8_qualification.py`。结果必须 PASS，且 `formal_matrix_authorized=false`。

## 6. 训练锚点

GPU0：A1；新进程 A2。  
GPU1：B1。

使用 `phase2_v8_training_anchor_worker.sh` 完整训练 64 步，并用 `phase2_v8_training_anchor_canary.sh` 生成 128 题 signatures。最后运行：

```bash
python scripts/finalize_phase2_v8_training_anchor_v2.py ...
```

所有 exact 和 absolute numeric gates 必须 PASS。ratio 是诊断项，不能成为单独误杀原因。

## 7. 生成 READY_FOR_HUMAN_REVIEW

运行：

```bash
python scripts/finalize_phase2_v8_release_go.py ...
```

输出必须为：

```text
status = READY_FOR_HUMAN_REVIEW
formal_matrix_authorized = false
required_human_authorization = START_PHASE2_V8_COMMON24
```

此处停止，提交所有 Q0/Q1/Q2 证据进行人工复审。

## 8. 人工授权后开始 24 格

人工确认后：

```bash
python scripts/authorize_phase2_v8_release.py \
  --ready READY_FOR_HUMAN_REVIEW.json \
  --operator-confirmation START_PHASE2_V8_COMMON24 \
  --output RELEASE_GO.json
```

两台机器分别运行：

```bash
bash scripts/phase2_v8_run_worker.sh ...
```

每台只运行自己的 12 格。不中途查看方法准确率，不改变参数，不按中间结果停止。工程失败立即停；科学结果好坏不影响继续完成冻结矩阵。

## 9. 长跑资源与恢复

每台建议：

- 1×4090D/4090；
- 8–12 vCPU；
- 32–64GB RAM；
- 150–300GB 本地 NVMe；
- 一个 GPU process；
- tmux/systemd 保活；
- 每 60 秒 heartbeat；
- CPU 打包可与下一格 GPU 运行重叠，但 backlog 不超过 1。

禁止 DDP、batch>1、vLLM、SGLang、FlashAttention 替换、TF32、`torch.compile`、运行中升级依赖。
