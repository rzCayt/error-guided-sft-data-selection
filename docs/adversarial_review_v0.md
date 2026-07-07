# Adversarial Review v0

Reviewer thread: `019f3b8b-a025-7fe3-9dca-434d1e78cfa8`

## Final Reviewer Findings

Blockers found:

- B128 matched-random baseline was invalid because targeted and matched-random shared 121 of 128 examples.
- Cross-split exact overlaps existed in the generated v0 data.
- The selection signal was too coarse to support strong claims because the simulated diagnostic and selector both leaned heavily on task and difficulty metadata.

Major concerns:

- Bias audit originally checked only stratum counts.
- Professor-facing wording could read as if a real small language model had already been evaluated.
- LoRA training path is currently a no-training evidence path, not a reproducible training run.
- Error profile documentation said error type was part of the profile, while the original profile was not grouped by error type.

## Fixes Applied In This Stage

- Added a per-stratum cap to error-guided selection so matched random can remain exact-stratum matched without overlapping the targeted subset in B128.
- Added `results/selection_bias_summary.csv` with overlap, answer-scale, reasoning-length, and marginal audit metrics.
- Added split-level leakage audit through `scripts/audit_splits.py`.
- Updated generator-wide `generate_all` to avoid exact prompt/parameter/answer duplicates across generated splits.
- Added `results/error_profile_by_error_type_v0.csv`.
- Updated README, project spec, professor summary, and selection policy to mark simulated diagnostics as placeholders.
- Added `docs/literature_review.md` and `docs/adversarial_review_protocol.md`.

## Remaining Risks

- The real model diagnostic still needs to replace simulated predictions.
- The selector is still primarily metadata-weighted; the next research iteration should add an error-type-aware variant or ablation.
- OOD template coverage is narrow and should be expanded before making robust generalization claims.
