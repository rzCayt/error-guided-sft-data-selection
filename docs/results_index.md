# Results index

## Primary chain

| Stage | Inputs | Main artifact | Decision |
|---|---|---|---|
| Initial model diagnostic | 100 development items | `results/real_base_diagnostic_summary.csv` | Establish an error profile only |
| Metadata selector audit | 500 candidates + frozen profile | `results/selector_identifiability_audit/summary.json` | Fail; do not train |
| Residual selector audit | 500 candidates + 100 diagnostics | `results/residual_selector_identifiability/summary.json` | Fail; candidate-specific signal absent |
| Model-aware F0/F1 | Qwen3-1.7B, selected final-layer gradients | `results/model_aware_signal_f0_f1/` | Engineering feasibility passes |
| Model-aware F2 | 8 candidates, 17 error and 8 correct queries | `results/model_aware_signal_f2/summary.json` | Scientific gate fails |
| Exact selector rerun | Frozen public inputs | `results/public_release_v1/selector_identifiability_rerun/summary.json` | Summary hash matches |
| Bounded model check | Qwen3-1.7B, first 25 mixed-family dev items | `results/public_release_v1/model_pipeline_check_25/` | Pipeline works; no effectiveness claim |

## Baseline table warning

`results/model_native_baseline_table/model_native_baseline_table.csv` contains
rows built from different input subsets and interface policies. It is an audit
table, not a model leaderboard. In particular:

- the older Qwen3-1.7B row uses a weighted-aggregation-only revised input set;
- the July reproduction check uses the first 25 mixed-family development items;
- the Qwen2.5-Math row reports an auxiliary math-native score, not the strict
  main score.

Numbers across these rows must not be interpreted as controlled model-scale
comparisons.
