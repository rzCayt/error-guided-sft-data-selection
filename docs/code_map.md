# Code map: trace claims to implementation

Use this map to navigate the project by scientific responsibility rather than
by file count.

| Responsibility | Primary implementation | Verification | Evidence |
|---|---|---|---|
| Synthetic task generation and gold answers | `src/eg_sft/data/generator.py`, `src/eg_sft/data/solver.py` | `tests/test_generation_and_selection.py`, `tests/test_solver.py` | `data/samples/` |
| Split and leakage controls | `scripts/audit_splits.py` | `tests/test_generation_and_selection.py` | generated split-audit outputs |
| Model interface and raw-output retention | `scripts/run_scale_model_diagnostic.py` | `tests/test_scale_model_diagnostic.py` | `results/public_release_v1/model_pipeline_check_25/` |
| Numeric parsing | `src/eg_sft/eval/parser.py` | parser cases in `tests/` | parser mode and prediction fields in raw JSONL |
| Original selector identifiability | `scripts/audit_selector_identifiability.py` | `tests/test_selector_identifiability_audit.py` | `results/selector_identifiability_audit/` |
| Residual selector identifiability | `scripts/audit_residual_selector_identifiability.py` | `tests/test_residual_selector_identifiability.py` | `results/residual_selector_identifiability/` |
| Model-aware engineering gate | `scripts/run_model_aware_f0_f1.py` | `tests/test_model_aware_f0_f1.py` | `results/model_aware_signal_f0_f1/` |
| Model-aware scientific gate | `scripts/run_model_aware_f2.py` | `tests/test_model_aware_f2.py` | `results/model_aware_signal_f2/summary.json` |
| Claim/artifact consistency | `scripts/validate_public_release.py` | `tests/test_validate_public_release.py` | `results/public_release_v1/manifest.json` |

## The shortest complete trace

For one row in the 25-item pipeline check:

1. Locate its `id` in `data/samples/dev_diagnostic.jsonl`.
2. Recompute its gold answer using the solver logic.
3. Locate the same `id` in the public raw-output JSONL.
4. Identify `raw_continuation`, `parser_mode`, `parsed_prediction`, and
   `numeric_accuracy`.
5. Recompute the aggregate count reported by
   `scale_model_diagnostic_summary.csv`.
6. Explain why that row validates the interface/parser chain but says nothing
   about selector or SFT effectiveness.

## The shortest selector trace

1. Read the four matching fields in the metadata-selector summary.
2. Locate where the same fields enter the non-random score.
3. Hold the exact stratum fixed and identify the remaining tie-breaker.
4. Explain why changing selected IDs across seeds is not evidence of
   candidate-level utility.
5. Confirm that the audit sets `training_allowed` to `false`.
