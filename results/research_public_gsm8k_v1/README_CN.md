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
- `diagnostic448_qwen2_5_1_5b_clean_34ad476/`：448 条 selection diagnostic
  的完整模型原始输出和运行证据。其中 349 条数值答对、99 条数值答错；
  两组都超过 64 条下限，因此后续可以按原计划构造 all-query 与
  error-conditioned 查询组。该分区不是 held-out test，也不能证明选择器有效。
- `query_groups_diagnostic448_clean_8b5273d/`：由干净代码从上述诊断结果生成的
  448 条 all-query 清单与 99 条 error-query 清单。只公开稳定 ID、源索引和哈希，
  不复制数据集原文；这只是后续表示和候选评分的冻结输入。
- `rds96_qwen2_5_1_5b_clean_1bd99f9/`：96 个分层 Tulu 候选在 all-query 与
  error-query 下的缩小规模 RDS+ 排名、训练长度审计和 10 候选重测清单。
- `utility10x2_qwen2_5_1_5b_clean_e214446/`：10 个候选、两个随机种子、每次
  一步 LoRA 更新后的 128 条验证集效用。ICC(A,1)=0.9729，通过 0.90 可靠性门槛；
  该结果只验证测量管线，不是正式 H1a 效应检验。

其他 smoke、旧版和独立重跑目录属于本地验证中间产物，已通过 `.gitignore`
排除，不属于规范发布结果。
