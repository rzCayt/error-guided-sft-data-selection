# Claim and evidence ledger

| ID | Claim | Verdict | Direct evidence | Missing evidence / boundary |
|---|---|---|---|---|
| C1 | The current metadata selector contains candidate-level signal beyond exact matching variables. | Not supported | `results/selector_identifiability_audit/summary.json` | Within-stratum ordering is only an ID/seed hash tie-breaker. |
| C2 | The residual operation-aware selector contains candidate-specific residual signal. | Not supported | `results/residual_selector_identifiability/summary.json` | Score is constant after full operation signature is fixed. |
| C3 | The model-aware score clears the preregistered tiny-pilot gate. | Not supported | `results/model_aware_signal_f2/summary.json` | Observed delta `0.019283` is below permutation p90 `0.024050`; only eight candidates were tested. |
| C4 | The offline metadata-selector audit is exactly reproducible. | Supported for the frozen inputs | `results/public_release_v1/selector_identifiability_rerun/summary.json` | Reproduction supports the audit implementation, not selection effectiveness. |
| C5 | The bounded Qwen3-1.7B output-parser-metric pipeline runs end to end. | Supported for 25 mixed-family dev items | `results/public_release_v1/model_pipeline_check_25/` | This is not a held-out evaluation and not evidence about SFT or selection. |
| C6 | Targeted selection improves LoRA/SFT over strong random baselines. | Untested | None | Requires a validated candidate-utility signal before training is authorized. |

## Allowed headline

> After controlling for model interface, parsing, task metadata, and static
> operation structure, the current selector has not demonstrated a
> candidate-level signal that justifies SFT escalation.

## Forbidden headlines

- Error-guided selection outperforms random selection.
- The proposed selector improves LoRA/SFT.
- The model-aware score predicts training utility.
- The result generalizes across tasks or models.

## Result-to-claim status

This ledger is based on deterministic artifact inspection. A formally
independent cross-model integrity review has not yet been run, so the ledger is
provisional at the paper-claim level even where artifact reproduction passes.
