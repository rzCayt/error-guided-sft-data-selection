# Adversarial Review Protocol

## Reviewer Thread

Thread title: `EG-SFT adversarial reviewer`

Thread id from the first implementation round:

```text
019f3b8b-a025-7fe3-9dca-434d1e78cfa8
```

The reviewer is read-only. It must not edit files, commit, push, train models, or change repo state.

## Review Cadence

Use stage-level review. Send a review package after each major stage, not after every small command.

Stages:

1. Repo/spec/documentation update.
2. Real base diagnostic.
3. B128 selection and bias audit.
4. LoRA smoke/full run.
5. Base/Random/Targeted comparison.
6. Professor-facing summary.

## Review Package Template

```text
Stage goal:
- ...

Changed files or planned changes:
- ...

Key outputs:
- ...

Known weaknesses:
- ...

Please return:
Blockers / Major concerns / Minor concerns / Required fixes / Verdict.
```

## Required Reviewer Verdicts

Every review must explicitly cover:

- test leakage risk
- matched-random fairness
- whether the selection signal is more than task/difficulty resampling
- placeholder or result-overclaiming risk
- whether the project can proceed to the next stage

## First Review Summary

The first reviewer pass raised these issues before the final report was complete:

- Matched-random implementation and documentation needed to state that exact stratum matching can require overlap with the targeted subset.
- Bias audit documentation claimed marginal and mean statistics, while the original output only reported stratum counts.
- Split seeds are independent, but narrow templates still require near-duplicate auditing before real test claims.
- Professor-facing wording should distinguish pipeline placeholders from real model results.
