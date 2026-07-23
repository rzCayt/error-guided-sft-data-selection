# 结果目录说明

- `artifact_reproduction_qwen3_4b/`：官方预计算绘图数据的固定 commit、哈希与
  GSM8K 预算 500 行提取。不是重新训练结果。
- `data_manifest_full_v2_fuzzy/`：当前规范数据版本，只包含源 ID、哈希、分区和
  泄漏审计，不重新分发 GSM8K 或 Tulu 原始文本。

其他 smoke、旧版和独立重跑目录属于本地验证中间产物，已通过 `.gitignore`
排除，不属于规范发布结果。
