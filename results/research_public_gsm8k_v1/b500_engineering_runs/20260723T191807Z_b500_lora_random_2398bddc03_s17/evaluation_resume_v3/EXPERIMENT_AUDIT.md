# Experiment Audit Report

**Date**: 2026-07-29
**Auditor**: isolated gpt-5.6-terra reviewer, xhigh reasoning, read-only
**Project**: random-500 LoRA engineering closure

## Overall Verdict: WARN

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS

The evaluator loads the pinned `openai/gsm8k` test split, validates
question and answer hashes, and scores model generations against the
dataset answer. No target is derived from model output.

Evidence:

- `scripts/resume_b500_gsm8k_eval.py:271`
- `scripts/resume_b500_gsm8k_eval.py:293`
- `scripts/resume_b500_gsm8k_eval.py:321`
- `src/eg_sft/evaluation/gsm8k_generation.py:50`

The reviewer independently matched all 1,319 frozen question and answer
hashes to the pinned cached dataset and exactly reproduced every saved
row score.

### B. Score Normalization: PASS

Metrics divide counts only by the number of examples. No metric uses
the model's own maximum, minimum, or mean as a denominator.

Independently recomputed values:

- numeric correct: 808/1,319
- numeric accuracy: 0.6125852918877938
- parse rate: 1.0
- strict parse rate: 0.489764973464746
- fallback-parsed rows: 673

Evidence: `src/eg_sft/evaluation/resumable.py:40`.

### C. Result Existence and Reproduction: WARN

All requested artifacts exist. The reviewer independently confirmed:

- 1,319 rows, 1,319 unique IDs, exact frozen-test order;
- raw-output SHA-256
  `b8ac83f17b868f9a3fd0c7ffda132a0ae7847909c5385538fa38d6d890230d21`;
- adapter SHA-256
  `a4231f7124ab5b225698218e43660c0083be4545ebe19d60fe7092ef426701fa`;
- 392 nonzero LoRA tensors and 18,464,768 adapter parameters;
- selection, selected-ID, run-config, semantic-evaluation, protocol,
  recipe, and thermal-policy hashes;
- saved-adapter reload with a nonzero active-versus-disabled logits
  difference.

Warnings:

1. The v3 manifest was created after inheriting a 48-row prefix. Its
   hash and contents are consistent, but the originating invocation
   manifest was not supplied.
2. The original monolithic process did not preserve its root metrics,
   token audits, raw outputs, or complete training telemetry after its
   evaluation was interrupted.
3. At review time, the audit implementation was untracked and its own
   code commit/hash was not written into `engineering_audit.json`.
4. `data_manifest.json` contains a stale absolute source config path,
   although the embedded config content matches.

### D. Dead Code Detection: WARN

All metric and audit functions used by the final resumable path are
called and produce saved artifacts. The original monolithic
`_evaluate_gsm8k` and validation-loss outputs have no preserved root
result because that invocation was interrupted.

### E. Scope Assessment: PASS

The demonstrated scope is one Qwen2.5-1.5B configuration, one
`random` B=500 selection, training seed 17, and one 1,319-example GSM8K
test evaluation. The saved claim boundary correctly states that this
does not compare selectors or estimate seed variance.

### F. Evaluation Type: real_gt

The evaluation uses externally supplied, pinned GSM8K test answers.

## Action Items

Integrity/provenance:

- Commit or content-hash the audit implementation and record its exact
  command and code identity in the audit artifact.
- Preserve any recoverable original training logs and explicitly mark
  unavailable telemetry as unavailable rather than reconstructing it.
- Bind every future raw-output file to a manifest before writing the
  first row.
- Record the inherited 48-row prefix as a bounded provenance gap; do
  not silently attribute it to the later v3 invocation.

Non-blocking:

- Keep fallback-inclusive accuracy separate from strict-marker parse
  rate.
- Treat the tokenizer regex warning as a portability warning; do not
  change the frozen tokenizer interface after seeing the result.
- Replace the stale absolute data-manifest path on the next clean
  artifact build.

## Claim Impact

- “The random-500 LoRA training/save/reload/1,319-example evaluation
  engineering loop completed”: **supported with provenance warnings**.
- “random, rds_all, and rds_error have been compared”:
  **unsupported**; only random/seed 17 exists.

Full reviewer trace:
`.aris/traces/experiment-audit/2026-07-29_run01/`.
