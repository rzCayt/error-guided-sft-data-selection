# 强力基线设计

## 结论

强力基线是 claim-bearing 阶段的必须项。没有通过 baseline audit，不能声称 `Targeted` 优于 `Random`，也不能把 LoRA 对比写成方法有效性证据。

它不是 `real_base_diagnostic` 的前置替代。真实 base diagnostic 仍然必须先产生，因为 error-guided selector、metadata-hard baseline 和后续审计都依赖真实 `dev_diagnostic` 错误画像。

进入 `selection_bias_audit` 后，baseline 构建必须显式使用真实 profile：

```powershell
python scripts/build_selection_sets.py --budget 128 --profile results/real_error_profile.csv --baseline-seeds 20260711,20260712,20260713
```

如果仍使用默认 `results/error_profile_v0.csv`，只能视为 simulated placeholder plumbing，不能进入 LoRA comparison。

## 主结论基线

主结论只能使用满足同预算、同候选池、同训练配置、同 parser、同 eval split 的 baseline。

当前最低基线套件：

1. `exact_matched_random_multi_seed`
   - 在 targeted subset 的 matching strata 内做 exact matched random。
   - 至少 3 个 seed，推荐 5 个 seed。
   - 报告 overlap、strata delta、marginal delta、均值和方差。
2. `stratified_random`
   - 按 candidate pool 的 task family、difficulty、answer magnitude、reasoning length strata 比例采样。
   - 用来检查 matched random 是否过度依赖 targeted distribution。
3. `metadata_hard_baseline`
   - 只使用 `dev_diagnostic` 聚合错误画像和 metadata 难度，不使用 test 信息。
   - 用来检查 error-guided selector 是否只是 hard-strata resampling。

## 辅助基线

这些 baseline 可以帮助解释结果，但不能单独支撑主结论：

- `BM25` 或 lexical similarity baseline：检查选择信号是否只是 surface-form similarity。
- full/same-budget sanity baseline：只有训练预算允许时运行。
- error-family hard subset baseline：等真实 error profile 稳定后再加入。

## Upper Bound

Oracle-style baseline 只能作为 analysis upper bound。它不能参与公平主结论，不能使用 test predictions、test labels、locked test metrics 或任何来自最终评估的信息。

## Baseline Audit

进入 LoRA comparison 前必须产出 baseline audit，至少包含：

- sample budget equality
- token/length budget check
- targeted/baseline overlap
- exact strata delta
- task family marginal delta
- difficulty marginal delta
- answer magnitude marginal delta
- reasoning length marginal delta
- multi-seed random variance
- prompt/parser/eval split/training config 一致性
- manifest：记录 `profile_path`、baseline seeds、输出文件和 placeholder 状态

如果 audit 显示 baseline 与 targeted 不可比，应把 comparison 标记为 invalid 或 exploratory，不能做有效性声明。
