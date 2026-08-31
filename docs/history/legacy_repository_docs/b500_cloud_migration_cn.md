> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# B=500 云端统一复现实验手册

更新时间：2026-08-23。

## 结论

主结果应在同一台云端 RTX 4090 上重新完成全部 9 项，而不是只补本机矩阵剩余的
7 项。本机已完成的 `random/17` 与 `rds_all/17` 来自 RTX 5060 Laptop；若只把
`rds_error/17` 放到云端，seed 17 的配对比较会混入硬件差异。

云端 9 项保持原模型、数据、三份选样清单、LoRA 配方、prompt、parser、种子、顺序
和停止门槛不变。唯一的新实验层面变化是：使用新的输出目录，并让全部 9 项共享同一
台 RTX 4090。原本机两项仅作工程闭环和跨硬件核对，不进入云端主表。

## 推荐实例

首选 AutoDL：

- 1 张独占 RTX 4090 24GB；
- 至少 8 vCPU、32GB 内存；
- 数据盘至少 150GB，把仓库与 Hugging Face 缓存都放在 `/root/autodl-tmp/`；
- 选择 PyTorch 2.8.0、Python 3.12、CUDA 12.8 镜像；
- 只选实时价格不高于 3 元/小时的实例；
- 本阶段总预算硬上限 80 元。

若 AutoDL 没有合适实例，备选 RunPod Secure Cloud RTX 4090。不要为这个 1.5B
LoRA 项目购买 A100、H100 或多卡实例。

## 计费与停止边界

- AutoDL 按量实例从开机到关机计费；浏览器关闭不会停止任务，也不会停止计费。
- 预计环境配置与下载 1–2 小时，9 项训练、评估和审计约 8–12 小时。
- 任何一次正式任务必须先通过 preflight；失败时不启动下一项。
- 单项运行完成后先生成审计并备份，再开始下一项。
- 累计费用达到 80 元、实例发生硬件错误、或者同一问题连续失败两次时停止。
- 9 项完成后关机；B=1000 与 4B 不属于本阶段。

## 首次环境配置

```bash
cd /root/autodl-tmp
git clone --branch research/cloud-b500-v1 \
  https://github.com/rzCayt/error-guided-sft-data-selection.git
cd error-guided-sft-data-selection

export HF_HOME=/root/autodl-tmp/hf-cache
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

python -m pip install -r requirements-cloud-b500.txt
python -m pytest -q
python scripts/preflight_b500_formal_matrix.py \
  --matrix-config configs/b500_formal_matrix_cloud_4090_v1.json
```

若访问 GitHub 或 Hugging Face 很慢，可在 AutoDL 中临时执行：

```bash
source /etc/network_turbo
```

下载结束后可取消代理：

```bash
unset http_proxy
unset https_proxy
```

## GPU 内核见证

该命令只做一个固定种子的 BF16 矩阵乘法，不训练模型：

```bash
python -c "import torch; torch.manual_seed(20260722); x=torch.randn(32,32,device='cuda',dtype=torch.bfloat16); y=x@x; print('WITNESS_OK', tuple(y.shape), y.dtype, torch.cuda.get_device_name(0), bool(torch.isfinite(y).all()))"
```

必须同时看到 `WITNESS_OK`、`(32, 32)`、`torch.bfloat16`、RTX 4090 和 `True`。

## 保存独立参考 tokenizer

```bash
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', revision='8faed761d45a263340a0528343f099c05c9a4323'); t.save_pretrained('/root/autodl-tmp/b500-reference-tokenizer')"
```

参考目录必须在仓库外，避免让正式运行的 Git 工作区变脏。

## 单项运行

第一项固定为：

```bash
tmux new -s b500_random_17
bash scripts/run_b500_cloud_one.sh random 17
```

断开 tmux：按 `Ctrl+B`，再按 `D`。重新进入：

```bash
tmux attach -t b500_random_17
```

包装脚本只运行用户明确指定的一项，不会自动开始下一项。正式顺序为：

1. `random/17`
2. `rds_all/17`
3. `rds_error/17`
4. `rds_all/29`
5. `rds_error/29`
6. `random/29`
7. `rds_error/41`
8. `random/41`
9. `rds_all/41`

## 每项完成标准

运行目录必须同时包含：

- `training_complete/training_metrics.json`
- `training_complete/adapter/adapter_model.safetensors`
- `evaluation/raw_outputs.jsonl`，严格 1,319 行
- `evaluation/metrics.json`
- `run_complete.json`
- `audits/formal_audit_v1.json` 及其 SHA-256 sidecar

审计命令中的 `RUN_DIR` 必须替换为刚完成任务的真实目录：

```bash
mkdir -p "$RUN_DIR/audits"
python scripts/audit_b500_formal_run.py \
  --matrix-config configs/b500_formal_matrix_cloud_4090_v1.json \
  --run-dir "$RUN_DIR" \
  --reference-tokenizer-dir /root/autodl-tmp/b500-reference-tokenizer \
  --output "$RUN_DIR/audits/formal_audit_v1.json"
```

只有审计状态为 `PASS`，才允许手动启动下一项。

## 研究结论边界

9 项完成前不得比较选择器有效性。完成后主结果是三随机种子的 GSM8K numeric exact
match；同时报告严格解析率、训练时间、有效训练词元、显存和失败案例。`rds_error`
相对 `rds_all` 的原冻结升级门槛仍为：平均提升至少 1.5 个百分点，且至少 2/3 个
配对种子同方向。门槛失败即停止，不新增 selector，不单独调参。
