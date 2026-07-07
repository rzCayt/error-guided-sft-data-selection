# Research Note: Zhen Liu-Oriented Version

## Angle

This project frames data curation as a post-training pipeline problem: diagnose a base model, convert its failures into a selection policy, and test whether targeted LoRA SFT beats a matched random baseline.

## Why It May Fit

- Clear post-training workflow rather than a one-off prompt benchmark.
- Data selection is measurable under fixed sample and token budgets.
- The pipeline can be extended to real model diagnostics, curriculum schedules, and active data acquisition.

## Current State

Implemented locally: deterministic data generation, solver labels, diagnostic error schema, targeted selection, matched random baseline, bias audit, and smoke scripts.

## Ask

Feedback on whether the diagnostic-to-curation loop is a useful RA direction, and what evidence threshold would make the first result credible.
