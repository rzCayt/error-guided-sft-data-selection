# 当前结果索引

本页只列当前公开阶段的规范产物，不把历史结果混进来。

| 产物 | 用途 | 身份检查 |
|---|---|---|
| [`main_results.json`](../../results/public_summary/main_results.json) | 规范机器可读结果与结论边界 | 由 `scripts/reproduce_public_summary.py` 重新计算 |
| [`main_results.csv`](../../results/public_summary/main_results.csv) | 扁平下游结果表 | 从规范 JSON 生成 |
| [`main_results_table.md`](../../results/public_summary/main_results_table.md) | 方便阅读的自动生成表格 | 从规范 JSON 生成 |
| [`experiment_registry.csv`](../../results/public_summary/experiment_registry.csv) | 24 个格子的 ID、方法、名单种子、训练种子和名单哈希 | 从冻结规范矩阵生成 |
| [`figures/manifest.json`](../../figures/manifest.json) | 图片来源与输出哈希 | 来源 SHA 必须与规范 JSON 一致 |
| [`configs/frozen/MANIFEST.json`](../../configs/frozen/MANIFEST.json) | 与运行配置逐字节相同的公开快照 | 每份快照必须与运行源一致 |
| [`claim_evidence_ledger.md`](claim_evidence_ledger.md) | 已支持、未支持和未测试的结论 | 每个数字都能定位到公开证据 |

构建规范结果所使用的证据文件哈希已经写入 `main_results.json`。运行 `python scripts/verify_public_release.py` 可一次检查全部身份。

