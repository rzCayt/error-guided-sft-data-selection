> **Historical snapshot.**
>
> This document records an earlier research stage and is not the current result.
>
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

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

## Public GSM8K v1 chain

| Stage | Inputs | Main artifact | Decision |
|---|---|---|---|
| Trainable RDS+ scoring | 448 queries, 99 errors, 96 Tulu candidates | `results/research_public_gsm8k_v1/rds96_trainable_qwen2_5_1_5b_clean_153363f/` | Freeze all-query and error-query scores |
| Utility measurement | 96 candidates, one fixed LoRA update each, 128 utility items | `results/research_public_gsm8k_v1/utility96_qwen2_5_1_5b_clean_3fdb8b5/` | Complete candidate-level utilities |
| Formal Tulu H1a | 96 utilities, two controls, 1,000 label permutations | `results/research_public_gsm8k_v1/h1a_formal_tulu96_clean_3fdb8b5/` | Pass all three preregistered gates; proceed to bounded training comparison |

The Tulu H1a decision is not a downstream SFT result. The GSM8K-domain
48-candidate boundary check and the `B=500` random/all-query/error-query
training comparison remain unfinished.

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
