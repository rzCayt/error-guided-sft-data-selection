# Adversarial Review Protocol

## Reviewer Thread

Thread title: `EG-SFT adversarial reviewer`

Thread id from the first implementation round:

```text
019f3b8b-a025-7fe3-9dca-434d1e78cfa8
```

The reviewer is read-only. It must not edit files, commit, push, train models, or change repo state.

The reviewer must output Chinese for all explanations and verdicts. The authoritative rubric is `docs/adversarial_review_rubric_cn.md`, and structured review responses should match `workflow/templates/review_response.json`.

## Review Cadence

Use stage-level review. Send a review package after each major stage, not after every small command.

This protocol is a fixed workflow for the project. Any future model diagnostic, selector change, LoRA run, result table, or professor-facing summary should go through this gate before being treated as credible project evidence.

Stages:

1. Repo/spec/documentation update.
2. Real base diagnostic.
3. B128 selection and bias audit.
4. LoRA smoke/full run.
5. Base/Random/Targeted comparison.
6. Professor-facing summary.

## Review Package Template

Use `workflow/templates/review_package.json` as the required package shape. Validate it before sending:

```powershell
python scripts/validate_workflow_packet.py --kind review_package --path workflow/templates/review_package.json
```

## Required Reviewer Verdicts

Every review must explicitly cover:

- test leakage risk
- matched-random fairness
- whether the selection signal is more than task/difficulty resampling
- placeholder or result-overclaiming risk
- whether the project can proceed to the next stage

Every review must also include:

- `阻塞项`
- `主要问题`
- `次要问题`
- `必修复`
- `评分表`
- `阶段判定`

## First Review Summary

The first reviewer pass raised these issues before the final report was complete:

- Matched-random implementation and documentation needed to state that exact stratum matching can require overlap with the targeted subset.
- Bias audit documentation claimed marginal and mean statistics, while the original output only reported stratum counts.
- Split seeds are independent, but narrow templates still require near-duplicate auditing before real test claims.
- Professor-facing wording should distinguish pipeline placeholders from real model results.
