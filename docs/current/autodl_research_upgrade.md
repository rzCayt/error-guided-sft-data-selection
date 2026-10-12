# AutoDL 部署：本地代码与资格测试交接

当前没有云实例，未租卡、未启动正式测量。部署包含公开代码、许可 fixture、冻结协议和哈希；不含模型权重、受限数据正文、凭据或旧适配器。CPU 随机小模型优化测试只证明代码路径，不能证明 Qwen 真实数学训练效果。

## 环境与部署包

目标为 Linux、Python 3.11/3.12、单张支持 BF16 的 GPU。显存和速度必须通过真实 canary（少量试跑）确定，不根据卡名承诺耗时。当前固定栈使用 TRL 0.26.2 与 Transformers 4.57.2，依据 [版本化 API](https://huggingface.co/docs/trl/v0.26.2/en/grpo_trainer)；升级版本须重新资格测试。

构建前完成仓库检查，再在仓库外创建一个全新输出目录：

```bash
python scripts/build_autodl_research_bundle.py --output-dir ../autodl-bundle-20261013
```

脚本重新验证发布清单、每个文件和打包后的 SHA-256，拒绝覆盖。解压到新的目录，不覆盖旧实验：

```bash
tar -xzf autodl-research-upgrade-20261013.tar.gz
cd research-upgrade
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/autodl-research.txt
python -m pip install -e .
python scripts/verify_research_upgrade.py
```

按当前 PyTorch 安装说明选择适合实例驱动的 CUDA wheel。先验证 `torch.cuda.is_available()`、BF16 和可见设备；只开放一张卡。不默认安装 vLLM、FlashAttention 或新教师服务，先确定朴素路径正确。

## SFT→GRPO 可行性闭环

准备已固定版本的 GSM8K。公开数据卡标记 [MIT 许可](https://huggingface.co/datasets/openai/gsm8k/blob/main/README.md)；源版本取自仓库现有配置。这里只用旧 development 分区，再固定分成 96 条训练和 32 条验证；不能把已用开发数据变成全新确认集。

```bash
python scripts/prepare_public_gsm8k_data.py --config configs/public_gsm8k_v1.json --output-dir work/data --skip-tulu
python scripts/run_reasoning_post_training.py --stage sft --config configs/reasoning_post_training_smoke.json --data-manifest-dir work/data --output-root work/runs
```

第一条只准备 GSM8K，无需获取 Tulu 或教师模型。SFT 完成后，使用打印出的实际目录作为 GRPO 父目录：

```bash
python scripts/run_reasoning_post_training.py --stage grpo --config configs/reasoning_post_training_smoke.json --data-manifest-dir work/data --output-root work/runs --parent-run work/runs/ACTUAL_SFT_RUN
```

运行前必须验证报价并登记预留预算。这些命令会在已有实例训练，不会自动租卡。默认 SFT 16 个优化步、GRPO 8 步、每题 4 个回答，是有限的工程资格测试。SFT 使用原回答监督组件；两阶段和前后贪心评测共用同一提示词与严格数值解析器。验证集与训练集 ID 分离；无法解析计错，不排除出分母。

GRPO 真正调用 `GRPOTrainer.train()`，继续已保存且指纹一致的 SFT 适配器。正确性奖励权重 1，格式奖励权重 0；不为更长回答奖励。β=0 明确关闭参考模型 KL 项，所以不能把 GRPO loss 当成 KL 或学会推理的证据。记录所有采样回答及奖励、有效奖励组、前后生成、环境版本、数据身份和保存后的适配器。所有组奖励相同或参数未改变时，资格测试失败。

保存训练 checkpoint 时不设置淘汰数量，也不清理旧目录。中断保留 `FAILED.json`、完整 traceback 和部分文件；本入口暂不提供已验证的自动恢复。GPU canary 必须检查保存/重新加载与恢复后下一步一致性，才允许扩展长训练；本地通过不替代这个门槛。

训练回调在步边界检查非有限值与墙钟超限；它不能终止卡死的下载、评测或单个算子，也不能停止实例计费。实例开机时间包含这些开销，操作方需设置实例时限并在保存输出后停止计费；本包不会擅自关机或删除文件。

## 复用 48×3 可靠性方案

先核验四个公开元数据：协议、48 候选面板、重合审计和适配器索引。此路线需要完整 GSM8K/Tulu 无正文清单：用同一准备命令但不加 `--skip-tulu`，输出到新的 `work/utility-data` 目录；遵守原第三方告知，不公开 Tulu 正文。旧适配器须另行取得并按索引校验。沿用现有预检和运行脚本，不另造测量逻辑。

先做仅合同的零状态 2 候选×3 seed 检查：

```bash
python scripts/run_candidate_utility_state_drift.py --protocol configs/frozen/candidate_utility_state_dependence_protocol_frozen_20260831_v3.json --preflight artifacts/state_dependence_preflight_v3.json --panel artifacts/state_dependence_candidate_panel48_v3.json --adapter-index artifacts/state_drift_adapter_evidence_index_v2.json --public-config configs/public_gsm8k_v1.json --data-manifest-dir work/utility-data --state-id zero_initialized_lora --zero-adapter-initialization-seed 17 --probe-seeds 17 29 41 --candidate-limit 2 --contract-only
```

准备全新资格目录、核定小批预算后，移除 `--contract-only` 并增加 `--run-dir work/q0` 才访问 GPU。Q0 用于运行/恢复资格测试，不能作为 48×3 可靠性结论。只有资格测试审计通过，才用新的目录执行完整 48×3，并调用原 `audit_candidate_utility_state_dependence_v3.py` 和 `analyze_state_dependence_u0_v3.py`。正式矩阵要求新增测量，不拼接历史值。U0 GO 才能比较四个最终状态；缺适配器就停。

## 成本预检

`scripts/plan_compute_batch.py` 输入实际时价、实测 profile 与单一账本，计入初始化、空闲、存储和耗时余量，超过阶段或总上限时退出。profile 必须包括：

```json
{
  "schema_version": "compute-profile-v1",
  "status": "not_measured",
  "gpu_model": "FILL_FROM_INSTANCE",
  "work_unit": "one candidate probe",
  "measured_at": "FILL_WITH_TIMEZONE",
  "measurement_sha256": "FILL_FROM_ACTUAL_LOG",
  "config_sha256": "FILL_FROM_ACTUAL_CONFIG",
  "wall_seconds_per_unit": null
}
```

这是不能通过预检的待填模板，不是假实测证据。第一小批资格测试以实际报价乘保守开机时限核定，后续批次才用其完整墙钟实测 profile。不得拿历史别的卡或不同配置的速度冒充当前实测。

```bash
python scripts/plan_compute_batch.py --ledger configs/research_upgrade_budget.json --profile work/profile.json --phase weeks_1_3 --batch-id u0a-YYYYMMDD --units 144 --hourly-cny ACTUAL_RATE --quote-at ACTUAL_TIMESTAMP_WITH_TIMEZONE --setup-hours MEASURED_SETUP --storage-cny ACTUAL_STORAGE --output work/u0a-cost.json
```

输出只是一份估计，`launch_authorized` 为 false。启动前重新核验 profile 对应同一卡、同一工作单位、同一代码/配置，登记该 batch 的 reserved 金额后串行执行；结算时释放预留并写实际 charged 金额。预算通过不代表实验或创新通过。
