# One-Page Professor Summary

## Project

**Error-Guided Data Selection for Data-Efficient LoRA SFT in Small Numerical Reasoning Language Models**

## Research Question

Can base-model diagnostic errors guide SFT data selection more effectively than matched random selection under the same sample and token budget?

## Pipeline

The project builds a controlled synthetic numerical-reasoning benchmark with deterministic solver labels. A base small language model is evaluated on a diagnostic split that is separate from training candidates and locked tests. Its failures are parsed into an error taxonomy covering arithmetic error, wrong formula, unit/scale error, temporal ordering, parse failure, and variable binding. The resulting error profile is used to select SFT examples from a separate candidate pool. A matched-random baseline is constructed to match task family, difficulty, answer magnitude, and reasoning length strata. If exact matching requires overlap with the targeted subset, the overlap is reported as a limitation rather than hidden. Locked ID and OOD tests are kept out of policy tuning.

## Current Evidence

The repository contains a reproducible generator, solver, simulated diagnostic path, selection policies, and bias audit. The simulated path is only a placeholder for local environments where model loading is unavailable; it validates the pipeline but does not support any claim of training gain.

## Next Step

Run `Qwen/Qwen2.5-0.5B` base diagnostic and a LoRA smoke test on a GPU machine, then compare Base, Random, and Targeted under B128/B256 budgets.

## Fit

The work is intended for RA conversations around post-training, data curation, reliable reasoning, and efficient small-model experimentation.
