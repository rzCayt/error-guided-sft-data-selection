# Fixed Project Workflow

This repository uses a stage-gated workflow. The goal is to keep the project credible enough for RA discussion by preventing silent leakage, weak baselines, and overclaimed results.

The authoritative workflow is now the Chinese, machine-checkable workflow:

- `docs/workflow_cn.md`: main-thread SOP and self-check rubric.
- `docs/adversarial_review_rubric_cn.md`: read-only reviewer rubric and required Chinese output format.
- `workflow/stages.json`: stage gates, required artifacts, checks, forbidden claims, and exit criteria.
- `workflow/templates/*.json`: structured stage plan, self-check, review package, and review response templates.
- `scripts/validate_workflow_packet.py`: local validator for workflow packets.

## Stage Gate

Each stage follows the same order:

1. Fill a structured stage plan.
2. Run or update the stage artifacts.
3. Fill a main-thread self-check.
4. Build a structured review package.
5. Validate the packet with `scripts/validate_workflow_packet.py`.
6. Send the package to the adversarial reviewer thread.
7. Fix blockers before moving to the next claim-bearing stage.
8. Commit and push only after verification and review pass.

## Adversarial Reviewer

Thread title: `EG-SFT adversarial reviewer`

Thread id:

```text
019f3b8b-a025-7fe3-9dca-434d1e78cfa8
```

The reviewer is read-only and must answer in Chinese. It should attack:

- test leakage and near-duplicate risk
- matched-random fairness
- whether selection is more than task/difficulty resampling
- simulated-result overclaiming
- LoRA reproducibility
- professor-facing contribution wording

## Required Review Package

Use the JSON structure in `workflow/templates/review_package.json`. The reviewer response must follow `workflow/templates/review_response.json` and the Chinese rubric in `docs/adversarial_review_rubric_cn.md`.

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
