# v3 Output Manifest

| Artifact | Purpose |
|---|---|
| `idea-stage/docs/research_contract.md` | 冻结研究问题、主张和AI贡献边界 |
| `refine-logs/EXPERIMENT_PLAN_V3_20260824.md` | Phase 0/1正式实验计划 |
| `refine-logs/EXPERIMENT_TRACKER_V3_20260824.md` | 实际实施状态和阻塞项 |
| `configs/budget_equivalent_v3_protocol.json` | 尚未绑定相似度/近重复簇SHA的准备协议 |
| `configs/budget_equivalent_lora_v3.json` | 500条、两次曝光、64步LoRA配方 |
| `src/eg_sft/selection/budget_equivalent.py` | 查询bootstrap、共同混合和MILP名单求解 |
| `src/eg_sft/training/token_budget.py` | 64步分区与曝光审计 |
| `scripts/prepare_budget_equivalent_inputs.py` | 云端表示导出和近重复簇生成 |
| `scripts/build_budget_equivalent_lists.py` | 16份名单和信息性闸门 |
| `scripts/run_budget_equivalent_cell_v3.py` | 单格公开运行入口 |
| `scripts/audit_budget_equivalent_cell.py` | 单格CPU只读审计 |

所有实际运行artifact位于`.aris/`，不可覆盖且不进入Git。
