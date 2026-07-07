# Contribution Statement for Professor Communication

## Short Version

I built a small research scaffold for studying whether base-model diagnostic errors can guide data selection for data-efficient LoRA SFT on numerical reasoning tasks. My contribution is the experimental design and reproducible pipeline, not a claim that the method already improves a model.

## What I Built

- A deterministic synthetic numerical-reasoning benchmark with solver-verified labels.
- Separate splits for candidate training data, base diagnostic profiling, ID testing, OOD template testing, and OOD range testing.
- A diagnostic pipeline that records predictions, parse success, numeric accuracy, output length, and error type.
- An error-guided data selector and a matched-random baseline.
- Bias and leakage audits, including split duplicate checks and targeted/matched overlap checks.
- A stage-level adversarial review workflow to catch weak baselines, leakage, and overclaimed results before moving to the next stage.
- Literature positioning around data-efficient instruction tuning, data selection, and LoRA.

## What I Did Not Claim

- I do not claim that error-guided selection already beats matched random.
- I do not claim real LoRA training results yet.
- I do not claim the synthetic tasks prove broad mathematical reasoning.
- I do not claim the current selector is final; the next version should add an error-type-aware selector and ablations.

## How To Explain My Role

Use this phrasing:

```text
I designed and implemented a controlled pilot pipeline for error-guided SFT data selection. The project focuses on experimental rigor: deterministic data generation, solver-verifiable labels, split discipline, matched-random baselines, leakage audits, and adversarial review before making performance claims. The current stage validates the research setup and prepares for real base-model diagnostic runs on Qwen2.5-0.5B. The next step is to replace simulated diagnostics with real model outputs and test whether diagnostic-error-aware selection adds signal beyond metadata-matched random sampling.
```

## Professor-Facing Emphasis

Emphasize:

- research question clarity
- controlled comparison design
- reproducibility
- awareness of baseline fairness
- willingness to report negative or inconclusive results
- readiness to run real base diagnostics and LoRA experiments

Avoid saying:

- "I built a financial LLM."
- "The model improved already."
- "The targeted method works."
- "The dataset proves reasoning ability."

## One-Minute Verbal Pitch

```text
I am building a small post-training research project on data-efficient SFT. The question is whether a base model's diagnostic errors can guide which SFT examples to train on, compared with a matched random baseline under the same budget. I built the generator, solver, split discipline, parser, error taxonomy, targeted and matched-random selection, and audit reports. A separate adversarial review thread checks for leakage, baseline unfairness, and overclaiming. The project is currently at the stage where the pipeline is ready for real Qwen2.5-0.5B base diagnostic, but I am not claiming training gains until the real diagnostic and LoRA comparison are run.
```

## If Asked "What Was Your Core Contribution?"

Answer:

```text
My core contribution is turning a broad RA idea into a reproducible experimental protocol: a solver-verifiable task setup, strict split discipline, diagnostic error profiling, fair matched-random comparison, and audit gates that prevent premature claims. The next research contribution will depend on whether real model diagnostics show that error-type-aware selection improves over metadata-matched random baselines.
```
