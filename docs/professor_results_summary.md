# Professor-facing evidence table

| Evidence | Scope | Result | What it supports | What it does not support |
|---|---:|---:|---|---|
| Qwen2.5-0.5B initial diagnostic | 100 dev items | 21% accuracy | A measurable failure profile exists | Selector or SFT effectiveness |
| Qwen3-4B model-native reference | 25 dev items | 84% strict accuracy | Model/interface choice materially affects the diagnostic | A cross-model causal conclusion |
| Qwen2.5-Math-1.5B auxiliary control | 25 dev items | 72% math-native auxiliary score | Strict formatting and mathematical competence must be separated | Main strict accuracy or selector quality |
| Metadata selector audit | 500 candidates; budget 128 | Fail; exact rerun hash match | Current non-random score is fully controlled by the matched baseline | Candidate-specific utility |
| Residual operation selector | 500 candidates | Fail | Static operation metadata still explains the score | Training effectiveness |
| Model-aware feasibility pilot | 8 candidates | Effect 0.0193 < permutation p90 0.0241 | The preregistered gate correctly stops escalation | Generalization or a reliable utility signal |
| July pipeline reproduction check | Qwen3-1.7B; 25 dev items | 19/25 accuracy; 25/25 parsed | Raw output → parser → metric chain runs end to end | Selector or SFT improvement |

## Frozen interpretation

The current error-guided selector has not demonstrated candidate-level signal
beyond simple matching variables. H1a candidate utility and any LoRA/SFT study
are future work and must not be described as completed.

## Primary artifacts

- `results/model_native_baseline_table/model_native_baseline_table.csv`
- `results/selector_identifiability_audit/summary.json`
- `results/residual_selector_identifiability/summary.json`
- `results/model_aware_signal_f2/summary.json`
- `results/public_release_v1/selector_identifiability_rerun/summary.json`
- `results/public_release_v1/model_pipeline_check_25/`
