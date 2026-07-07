# Fixed Project Workflow

This repository uses a stage-gated workflow. The goal is to keep the project credible enough for RA discussion by preventing silent leakage, weak baselines, and overclaimed results.

## Stage Gate

Each stage follows the same order:

1. State the stage goal.
2. Run or update the stage artifacts.
3. Run local verification.
4. Send a review package to the adversarial reviewer thread.
5. Fix blockers before moving to the next claim-bearing stage.
6. Commit and push only after verification passes.

## Adversarial Reviewer

Thread title: `EG-SFT adversarial reviewer`

Thread id:

```text
019f3b8b-a025-7fe3-9dca-434d1e78cfa8
```

The reviewer is read-only. It should attack:

- test leakage and near-duplicate risk
- matched-random fairness
- whether selection is more than task/difficulty resampling
- simulated-result overclaiming
- LoRA reproducibility
- professor-facing contribution wording

## Required Review Package

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

## Current Gate Status

Allowed next step:

- Real base diagnostic as exploratory data collection.

Not allowed yet:

- Claim-bearing Random vs Targeted LoRA comparison.
- Professor-facing performance claims.

Required before LoRA comparison:

- Run split leakage audit.
- Run selection bias audit for the exact budget.
- Require `overlap_rate=0` or explicitly mark the comparison invalid.
- Add or ablate an error-type-aware selector.
- Replace simulated diagnostic rows with real model raw outputs and run metadata.
- Store Qwen smoke evidence as a machine-readable artifact, not only prose.
