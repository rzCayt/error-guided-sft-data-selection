# 结果目录说明

- `artifact_reproduction_qwen3_4b/`：官方预计算绘图数据的固定 commit、哈希与
  GSM8K 预算 500 行提取。不是重新训练结果。
- `data_manifest_full_v2_fuzzy/`：当前规范数据版本，只包含源 ID、哈希、分区和
  泄漏审计，不重新分发 GSM8K 或 Tulu 原始文本。
- `overfit16_qwen2_5_1_5b_clean_a840ef2/`：Qwen2.5-1.5B 的 16 样本 LoRA
  过拟合指标、选中 ID、运行 manifest 和 adapter 哈希。adapter 权重只保存在
  本地 checkpoint 目录，不提交到 Git。
- `calibration64_qwen2_5_1_5b_clean_7af04b1/`：64 条 interface calibration
  的完整模型原始输出、严格解析结果、fallback 解析模式、数值正确性和运行证据。
  该分区只用于冻结接口，不是最终模型准确率。

其他 smoke、旧版和独立重跑目录属于本地验证中间产物，已通过 `.gitignore`
排除，不属于规范发布结果。
