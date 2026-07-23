# Public GSM8K v1：第一阶段进展

更新时间：2026-07-23。

## 已经完成

### 1. 官方产物复现

官方仓库固定在：

`c1dc0286b06b9e2a857d925e516938f4c9619dc2`

官方 Qwen3-4B 绘图脚本已在本机成功生成六张 PDF。自动审计脚本同时保存六个
源 CSV 的 SHA-256，并提取 GSM8K、预算 500、三个种子的固定结果。

| 方法 | 官方预计算 exact match 均值 |
|---|---:|
| Random | 82.3604 |
| RDS+ (RR) | 80.6419 |
| LESS (RR) | 82.7394 |

这些数字只是官方预计算 CSV 的可复核提取，不是本项目重新训练 Qwen3-4B 得到的
结果。

### 2. 数据版本冻结

| 对象 | 固定 revision |
|---|---|
| GSM8K | `740312add88f781978c0658806c59bc2815b9866` |
| Tulu v2 processed | `e217e5f72f7a0d10748d4c61abc5856338d90c7f` |
| Qwen2.5-1.5B Base | `8faed761d45a263340a0528343f099c05c9a4323` |

GSM8K 的数据卡声明 MIT 许可证，Qwen2.5-1.5B 声明 Apache-2.0。当前 Tulu
processed 数据卡没有明确许可证字段，因此公开结果只保存源 ID、哈希和统计，不
重新分发原始文本。

### 3. GSM8K 固定划分

| 协议分区 | 数量 |
|---|---:|
| interface calibration | 64 |
| selection diagnostic | 448 |
| candidate utility validation | 128 |
| development | 128 |
| in-domain candidate pool | 6,705 |
| held-out test | 1,319 |

训练题共 7,473 条，测试题共 1,319 条。规范化问题哈希的训练—测试交集为 0。

### 4. Tulu 10K 候选池

- 原始行数：197,196；
- 无效对话结构：61；
- 精确去重后 prompt 数：194,230；
- 最终候选数：10,000；
- 候选 ID 唯一数：10,000；
- prompt 哈希唯一数：10,000；
- 精确 GSM8K user-prompt 重合：0；
- 在候选池填满前，5-gram 模糊检查排除 97 条高重合候选。

独立重跑后，下列三个文件逐字节一致：

| 文件 | SHA-256 |
|---|---|
| `gsm8k_records.jsonl` | `888669bd443e688993c51488e9e0bca2e460df7b53ac490c2d4e0836bbca6be7` |
| `tulu_candidate_pool.jsonl` | `3c461693dcf1d7dafb8c40fa10a43dd63bedbdcd8f802ef8ad8ca29d6f4190e7` |
| `data_manifest.json` | `b4779f7010dfb666e7876f25e16d539c8b0df72888dd9567090d1827b5911ef4` |

### 5. 训练前代码契约

新增测试覆盖：

- GSM8K 标准答案和模型输出的严格数值解析；
- 缺标记、多标记、公式、单位、占位符和尾随文本拒绝；
- response-only label mask；
- padding label mask；
- 只有 LoRA 参数可以训练；
- 冻结参数不能获得梯度或被优化器修改；
- LoRA adapter 保存和加载后输出一致；
- 运行目录不可覆盖、配置哈希与键顺序无关。

全仓测试结果：`125 passed`。

### 6. Qwen2.5-1.5B 的 16 样本 LoRA 过拟合

规范运行对应干净 commit：

`a840ef25bebceec567e5f883b0a02ec78533a30d`

manifest 记录 `git_is_dirty=false`。训练使用 16 条固定 development 样本，没有
接触 selection diagnostic、utility validation 或 held-out test。

| 指标 | 结果 |
|---|---:|
| 训练前 token loss | 0.548746 |
| 训练后 token loss | 0.003192 |
| loss 降低比例 | 99.418% |
| 优化步数 | 32 |
| 训练耗时 | 55.12 秒 |
| 监督 token / 秒 | 281.15 |
| 峰值显存 | 3.742 GiB |
| LoRA 可训练参数 | 18,464,768 |
| 可训练参数比例 | 1.182% |

保存 adapter 后释放原模型，重新加载基础模型和 adapter，再次计算得到的 loss 仍为
`0.0031916847242529`，与保存前绝对差为 0。

adapter 权重 SHA-256：

`f45cf7c93d288a3a1a4209c27a5ee7a5b21ad5e76c30df3d74b330ac8c47807d`

这证明训练、梯度隔离、保存和加载管线可以工作，但不证明泛化能力，也不证明任何
数据选择器有效。

## 当前还不能声称什么

- 尚未重新训练 Qwen3-4B；
- 尚未生成 448 条 diagnostic 的基础模型答对/答错标签；
- 尚未完成候选效用 H1a；
- 尚未证明 error-conditioned RDS+ 超过 all-query RDS+ 或随机选择。

## 下一段代码目标

1. 固定 GSM8K prompt、生成参数和 raw-output schema；
2. 在 interface calibration 64 条上冻结解析与生成接口；
3. 开始 448 条 diagnostic 的基础模型推理；
4. 根据答对/答错结果生成 all-query 与 error-conditioned 查询表示；
5. 实现 H1a 的单候选微更新和 utility loss 计算。
