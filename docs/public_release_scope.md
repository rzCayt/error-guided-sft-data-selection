# Public release scope

## Purpose

The professor-facing release is a compact research audit, not a dump of every
exploratory run. Inclusion is based on three questions:

1. Does the artifact directly support a bounded project claim?
2. Can a reader reproduce or mechanically inspect it from public inputs?
3. Can it be included without exposing local paths or implying that unfinished
   work is complete?

## Include in the first release

### Core code

- Data generation, split audit, solver, parser, and evaluation modules.
- `scripts/audit_selector_identifiability.py`
- `scripts/audit_residual_selector_identifiability.py`
- `scripts/run_model_aware_f0_f1.py`
- `scripts/run_model_aware_f2.py`
- `scripts/run_scale_model_diagnostic.py`
- `scripts/build_model_native_baseline_table.py`
- Tests corresponding to the code above.

### Core evidence

- `results/selector_identifiability_audit/`
- `results/residual_selector_identifiability/`
- `results/model_aware_signal_f0_f1/`
- `results/model_aware_signal_f2/`
- `results/model_native_baseline_table/`
- Sanitized artifacts under `results/public_release_v1/`, built from the original
  records in `results/professor_package_validation/`

### Core explanation

- `README.md`
- `docs/professor_research_note_en.md`
- `docs/claim_evidence_ledger.md`
- `docs/results_index.md`
- `docs/reproducibility.md`
- `docs/code_takeover_guide.md`

## Keep local for now

The following artifact families remain useful as a research log but are not
needed for the first professor-facing release:

- early answer-only, chat-vs-completion, wording, prompt-rescue, and parser
  iteration runs;
- diagnostic-query-bank environment recovery and model-load troubleshooting;
- WSL/cache/snapshot recovery designs containing machine-specific paths;
- workflow packets and long-form stage orchestration state;
- failed or incomplete model-loading checks;
- CV, email, slides, and application trackers.

Keeping these artifacts local is not a claim that they are wrong. It prevents
the release from confusing process history with the final evidence chain. No
local artifact is deleted by this classification.

## Explicitly unfinished

- H1a candidate-utility validation.
- Independent target-model diagnostic replicates.
- Any LoRA/SFT comparison matrix.
- Financial-domain LLM extension.
- A paper-level generalization claim.

These items may be proposed as next work, but must not appear as completed
contributions.
