# Phase 2 v8.2 最终对抗审计与启动裁决

日期：2026-08-28  
审计对象：`phase2_v8_gptpro_review(2).zip` 及其实际源码、配置、manifest、测试、物化输入合同与运行脚本。  
审计身份：严格论文审稿人、复现实验工程师、RA 导师。

## 一、最终裁决

### 科学设计

**保留 24-cell 主矩阵，不推倒重来。**

主矩阵仍为：

```text
2 methods
× 4 frozen list replicates
× 3 training seeds {17, 29, 41}
= 24 all-new common-mix cells
```

两个方法：

- `random_common_mix`
- `rds_error_common_mix`

历史环境中的 seed17 结果不进入 v8 主分析，只作为外部历史证据。新矩阵在同一代码时期、同一模型快照、同型号 4090D、同一训练与评估后端中重新运行全部三个种子，因此避免了“训练 seed 与硬件/代码时期共线”。

### 工程启动状态

```text
本地源码/静态审计：PASS（受当前容器依赖限制）
正式 AutoDL Q0：尚未运行
GPU qualification：尚未运行
正式 24-cell matrix：尚未授权
```

因此，本包达到的是：

> **可以部署到 AutoDL 并开始 Q0、离线数据资格和 GPU qualification；尚不能跳过资格闸门直接开始 24 格。**

只有 Q0、Q1、Q2 全部 PASS，并生成 `READY_FOR_HUMAN_REVIEW.json` 后，人工发送 `START_PHASE2_V8_COMMON24`，正式矩阵才可以开始。

## 二、本轮发现并修复的六个实质问题

### P0-1：冻结权威仍可被替换

原实现允许 Python 入口接收另一份格式合法但并非 release 权威的 canonical manifest；Q0 与最终 GO 也可能在不同 canonical 文件上分别通过。

#### 风险

- 旧统计协议、旧训练锚点或旧矩阵被替换；
- README 声称冻结，但实际运行入口仍可绕过；
- Q0 通过的对象与正式长跑对象不一致。

#### 修复

- 唯一允许的 authority 固定为：
  `configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json`；
- canonical role 集合必须精确等于冻结的 12 个角色；
- `finalize_phase2_v8_release_go.py` 必须证明 Q0、matrix、statistics、canary、training anchor、semantic code 和 deployment 均属于同一冻结 release；
- 任何命令行传入的替代 canonical 文件都 fail closed。

### P0-2：训练锚点的 cross/same 漂移比可能误杀

原训练锚点同时设置绝对数值门槛和：

```text
cross-GPU drift <= 5 × same-GPU drift
```

若同卡 A1/A2 完全一致或漂移接近 0，即使跨卡差异远小于冻结绝对阈值，比例也会趋于无穷并错误 FAIL。

#### 修复

- 保留冻结的绝对门槛与 128 题 token-exact signature 作为权威闸门；
- cross/same ratio 只保留为 diagnostic context；
- same-GPU drift 小于预注册 floor 时不计算比例，不允许它覆盖绝对门槛结论；
- 阈值必须在看见 GPU 结果前冻结，anchor 失败后不得临时放宽。

### P0-3：训练 seed 在主统计中的角色自相矛盾

原协议写明 training seeds 是 fixed blocks，但 primary bootstrap 又把仅有的三个 seed 当作总体样本重采样。三个 cluster 无法稳定支持“任意训练 seed 总体”的非参数推断。

#### 修复

主分析改为：

- 固定观察到的 seed17、29、41 三个 block；
- 分别报告每个 seed 的 method effect；
- 完整 leave-one-seed-out；
- Random 和 RDS 的 list replicate 分别重采样；
- 同一 bootstrap draw 中，题目抽样在两种方法间共享；
- OOD 采用 SVAMP、ASDiv、MultiArith 数据集等权 macro，不池化 3841 道题。

将“对 seed 总体进行重采样”的版本降级为 exploratory sensitivity。

同时，±1pp 等效不允许成为 primary claim；即便 90% CI 落入区间，也只能标为 `exploratory_equivalence_signal_only`。

### P0-4：正式长跑前未证明四个数据集都能离线读取

worker 冻结 `HF_DATASETS_OFFLINE=1`。原 canary 主要覆盖 GSM8K，但没有在两个 fresh host 上证明 SVAMP、ASDiv 与 MultiArith 的 pinned revision 已完整缓存并与冻结 records 一致。

#### 风险

某个 cell 训练和 GSM8K 完成后，第一次进入 OOD 才因 cache 缺失失败，造成昂贵而可避免的中断。

#### 修复

新增：

```text
scripts/qualify_phase2_v8_offline_datasets.py
```

它必须在 `HF_DATASETS_OFFLINE=1` 下：

1. 加载 GSM8K train/test；
2. 加载 SVAMP、ASDiv、MultiArith 的 pinned revision；
3. 用冻结 source index、question/answer hash 或 row hash 逐条核验；
4. 生成稳定、无原始文本的 dataset-cache contract；
5. 将 contract SHA 写入 environment manifest；
6. 两台机器 environment equality 要求该 SHA 完全相同。

### P0-5：环境合同只冻结了部分 Python 依赖

原 environment manifest 固定了 Torch、Transformers、PEFT、Accelerate 和 Datasets，但遗漏了会影响 tokenizer、Arrow 数据读取、adapter 序列化和数值路径的依赖。

#### 修复

两台机器现在必须同时匹配：

```text
safetensors 0.7.0
numpy 2.3.2
huggingface-hub 0.36.0
tokenizers 0.22.1
pyarrow 22.0.0
fsspec 2025.9.0
dill 0.4.0
multiprocess 0.70.18
```

这些版本与原已冻结的 Python、Torch、CUDA、Transformers、PEFT、Accelerate、Datasets、driver、model/tokenizer tree 共同进入 environment contract。正式 worker 进程启动前也显式设置 Python hash、CUBLAS 和 offline 环境。

### P0-6：物化训练输入目录仍可被替换

原 worker 和 training-anchor 只检查传入目录中的 `MATERIALIZATION_COMPLETE.json` 具有 PASS、24 格和相同 matrix SHA，却没有要求它就是 canonical runtime 中冻结的物化合同。理论上，另一份伪造或过期目录可以通过浅层检查。

#### 修复

- CPU preflight、training anchor 和正式 worker 都要求：

```text
TRAINING_INPUT_ROOT/MATERIALIZATION_COMPLETE.json
== canonical role: materialized_contracts
```

- 文件路径和 SHA 均须与唯一 canonical authority 一致；
- 三个 training anchor 的 completion 报告记录相同的 canonical runtime SHA 与 materialized-contract SHA；
- 最终 `READY_FOR_HUMAN_REVIEW` 再次核对这些 binding；
- materialized v4 已证明 24/24 cell 的选样、token、label、训练配置在同一 method×list 跨 seed 完全不变，只有 seed 驱动顺序、step plan 与 RNG 不同。

## 三、仍然存在但不阻塞本轮的研究限制

1. **四份 RDS 名单不是四个完全独立政策。** 名单高重合会降低有效重复数；论文必须报告 Jaccard、rank correlation 和 overlap-adjusted sensitivity。
2. **24 格不是“证明等效”的设计。** 它主要估计方向、seed 翻转、list 变化和结果稳健性；±1pp 等效功效不足时必须写 `inconclusive`。
3. **Random replicate 与 RDS replicate 编号不构成天然配对。** 主 bootstrap 分别在两个 selector 内重采样名单，不能强制 rep1 对 rep1。
4. **本轮仍是单模型、单预算、单候选池、单 selector family。** 结果可作为 RA 级 clean replication 和论文第一张可靠性表，不能单独支撑顶会一般性主张。
5. **common-mix estimand 是共同可行域中的残余排序价值。** 它不是 free-mix 的“控制来源/长度后的直接因果效应”。

## 四、启动标准

### Q0：CPU release gate

必须全部 PASS：

- fresh extract 的 deployment manifest 0 missing / 0 mismatch；
- 唯一 canonical authority；
- clean Git commit；
- 全量 pytest；
- v8 targeted tests；
- semantic-code lint；
- failure injection；
- 24/24 contract-only；
- 物化输入合同审计；
- release archive SHA 绑定。

### Host data qualification

fresh host 无 cache 时，先在正式资格之外按 pinned revision 运行 `stage_phase2_v8_offline_datasets.py`；随后关闭网络依赖。预置报告不属于 PASS 证据。

每台机器必须在离线模式通过四个数据集资格测试，并产生：

```text
HOST_PREFLIGHT_ROOT/dataset_cache_qualification.json
```

### Q1：推理桥接

两张卡必须：

- 同型号 RTX 4090D/4090；
- 不同 GPU UUID；
- 环境合同完全一致；
- base16 跨卡逐层签名一致；
- archived adapter128 新 block token-exact；
- 历史 adapter 只声明 semantic bridge，不冒充历史 token exact；
- batch>1 仍禁止。

### Q2：训练锚点

运行：

```text
GPU0: A1, fresh process A2
GPU1: B1
```

必须满足：

- 训练输入合同和 64-step token plan 完全一致；
- instantaneous loss vector 在冻结绝对门槛内；
- adapter tensor 绝对差、余弦和 norm 差在冻结门槛内；
- A1/A2 与 A1/B1 的 128 题 signature 均完全一致；
- 历史 adapter semantic bridge 通过；
- ratio 只作诊断，不作独立 hard gate。

### 人工放行

Q0–Q2 通过后只能生成：

```text
status = READY_FOR_HUMAN_REVIEW
formal_matrix_authorized = false
```

人工核对 binding 后发送：

```text
START_PHASE2_V8_COMMON24
```

才能生成 `RELEASE_GO.json` 并启动 worker。

## 五、最终 GO / NO-GO

```text
科学矩阵设计：GO，保持 24 格不变
修订源码与协议：GO，可进入 AutoDL Q0
GPU qualification：待真实机器执行
正式 24 格：CONDITIONAL NO-GO，直到 Q0–Q2 和人工放行完成
```

本轮不再增加 cell，不改 selector，不改模型，不改预算。若资格测试失败，先修环境或实现；不得以“让实验跑起来”为理由放宽科学阈值。
