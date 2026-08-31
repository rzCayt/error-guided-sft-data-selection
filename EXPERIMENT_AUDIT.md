# Experiment Integrity Audit

**Date:** 2026-08-31  
**Auditor:** Internal checklist fallback using the `experiment-audit` A–F contract  
**Independence:** No. The required external reviewer backend was unavailable, so this report must not be described as a cross-model independent audit.  
**Project:** LLM Post-training Data Selection: A Controlled Study

## Overall verdict: WARN

## Integrity status: warn

No fake ground truth, self-normalized accuracy, phantom headline number, or dead reported metric path was found. The warning is retained because the public repository includes audited aggregate evidence rather than all 24 cells' raw generations and adapter artifacts, and because the empirical scope is one model scale, one main budget, one candidate pool, and arithmetic-focused tasks.

## A. Ground-truth provenance: PASS

- GSM8K scoring parses the dataset-provided answer and compares it directly with the parsed model prediction in `src/eg_sft/evaluation/gsm8k_generation.py:42-83`.
- The public GSM8K data layer verifies each source answer against its frozen SHA in `src/eg_sft/data/public_gsm8k.py:34-40`.
- OOD records are constructed from upstream answer fields, store only hashes and a numeric gold value, and reject ambiguous golds in `src/eg_sft/evaluation/arithmetic_ood.py:17-65`.
- At execution time, OOD source rows are revalidated against source-row, question, answer, and gold hashes in `src/eg_sft/experiment/budget_equivalent_ood_runtime.py:96-125`.

The main accuracy evaluations are classified as `real_gt`. The earlier candidate-utility experiment uses independent gold-response loss reduction as an explicitly named utility proxy; it is not reported as task accuracy.

## B. Score normalization: PASS

- GSM8K correctness is a direct numeric equality comparison in `src/eg_sft/evaluation/gsm8k_generation.py:50-83`.
- OOD correctness is a direct numeric equality comparison in `src/eg_sft/evaluation/arithmetic_ood.py:69-93`.
- OOD accuracy and parse rates are counts divided by the fixed number of evaluated records in `src/eg_sft/experiment/budget_equivalent_ood_runtime.py:192-208`.
- The Phase 2 primary analysis averages binary item vectors and bootstraps list, fixed seed blocks, and items; it does not divide a score by the model's own maximum (`src/eg_sft/experiment/phase2_v8_statistics.py:213-246`).

No reported accuracy uses prediction-derived max/min normalization.

## C. Result-file existence and number matching: PASS with public-scope warning

- `python scripts/reproduce_public_summary.py --check` recomputes three public outputs from nine SHA-addressed inputs and passes.
- `python scripts/verify_public_release.py` verifies all nine evidence byte counts and SHA-256 values.
- The headline GSM8K and OOD values in both README files match `results/public_summary/main_results.json` exactly.
- `results/public_summary/experiment_registry.csv` contains the 24 cells from the frozen canonical matrix.

**Warning:** the public repository deliberately excludes large raw generations and adapter weights. Therefore a reader can verify the aggregate evidence identities and analysis code, but cannot reconstruct every cell metric from raw text using Git alone. This is stated in `REPRODUCE.md` and must remain visible.

## D. Dead-code detection for reported metrics: PASS

- `score_generation` is called by formal generation workers and read-only audit scripts, including `scripts/run_cloud_v2_formal_eval_worker.py:250` and `scripts/audit_b500_formal_run.py:379`.
- `score_ood_generation` is called by `scripts/run_budget_equivalent_ood_eval_worker.py:326`.
- `audit_complete_dataset` recomputes formal OOD metrics and is called by `src/eg_sft/experiment/budget_equivalent_ood_audit_v2.py:40`.
- `fixed_seed_common_bootstrap` and `crossed_common_bootstrap` are called by the final unblinded analysis in `scripts/analyze_phase2_v8_unblinded.py:60-83`.

This statement applies to the current reported metric path, not to every historical helper retained in the repository.

## E. Scope assessment: WARN

Actual scope:

- one base model and scale: Qwen2.5-1.5B Base;
- one main 500-example supervision regime;
- two compared policies;
- four list constructions per method;
- three training seeds;
- GSM8K plus three arithmetic OOD tasks.

The current README uses “controlled study” and “pilot,” explicitly limits generalization, and does not use “comprehensive,” “extensive,” or “robust” to describe empirical coverage. The scope is sufficient for a controlled pilot, not for a general selector verdict.

## F. Evaluation classification

| Evaluation | Classification | Reason |
|---|---|---|
| GSM8K numeric accuracy | `real_gt` | Dataset-provided answer parsed and compared with prediction |
| SVAMP / ASDiv numeric / MultiArith | `real_gt` | Upstream answer fields validated and hashed |
| Strict-format rate | `real_gt` for the formatting contract, secondary behavioral metric | Deterministic parser rule, not a reasoning-accuracy substitute |
| Candidate one-step utility | `supervised_proxy` | Independent gold-response NLL reduction, explicitly reported as utility rather than accuracy |
| Synthetic selector stage | `simulation_only` | Historical diagnostic only; excluded from the current main result |

## Action items

1. Keep the raw-generation limitation visible in the reproduction guide.
2. Do not describe this report as independent or cross-model until an authorized external backend reviews the same public paths.
3. Keep State Dependence v3 labeled unrun until GPU qualification and formal measurements exist.
4. Preserve the current single-source result checks and release manifest in CI.
5. If a paper draft is prepared, supply raw-cell evidence through an appropriate archival repository or documented access process rather than implying Git-only raw reproducibility.

## Claim impact

- **C1 — 24-cell pilot found no reliable RDS advantage:** supported within the stated regime; integrity status WARN due public raw-artifact omission.
- **C2 — RDS is generally ineffective:** unsupported.
- **C3 — RDS and Random are equivalent:** unsupported.
- **C4 — Tulu96 passed its original candidate-utility gate:** supported for that frozen proxy experiment; does not imply set-level gain.
- **C5 — state dependence exists:** untested and prohibited.

