# Project Spec

## Title

Error-Guided Data Selection for Data-Efficient LoRA SFT in Small Numerical Reasoning Language Models

## Research Question

Can base-model diagnostic errors guide SFT data selection more effectively than matched random selection under the same sample and token budget?

## Hypothesis

When the base model has a measurable diagnostic weakness, selecting candidate-pool examples that target the same task family, difficulty, answer scale, and likely error mode should improve locked numerical-reasoning evaluation more efficiently than a random sample matched only on coarse metadata.

## Non-Goals

- This is not a finance LLM project.
- This is not financial QA or investment advice.
- Finance-style wording is only a controlled shell for numerical reasoning.
- The project does not claim production training results until real LoRA runs are completed and logged.

## Splits

| Split | Purpose | Used For Selection? | Used For Evaluation? |
|---|---:|---:|---:|
| `candidate_pool` | Training sample reservoir | Yes | No |
| `dev_diagnostic` | Base-model error profiling | Yes, through aggregate profile only | No |
| `test_id` | Locked in-distribution evaluation | No | Yes |
| `test_ood_template` | Locked template generalization | No | Yes |
| `test_ood_range` | Locked numeric range generalization | No | Yes |

Selection policies must not inspect test predictions, test labels beyond split construction metadata, or test metrics.

## Pipeline

1. Generate deterministic synthetic examples with solver-verified answers.
2. Run base diagnostic on `dev_diagnostic`.
3. Parse model answers and map failures into the error taxonomy.
4. Build an error profile by task family, difficulty, magnitude, reasoning length, and error type.
5. Select a targeted subset from `candidate_pool`.
6. Select matched random subsets with the same budget and matching buckets.
7. Run LoRA SFT smoke/full experiments when hardware and packages are available.
8. Evaluate Base, Random, and Targeted on locked tests.

## Metrics

- Parse success rate.
- Exact numeric accuracy with tolerance.
- Mean absolute percentage error where applicable.
- Accuracy by task family and difficulty.
- Error-type distribution.
- Token/sample budget compliance.

## Initial Model Choice

Start with `Qwen/Qwen2.5-0.5B` or another small open model of similar size. If local hardware cannot load the model, keep the model interface and emit a no-training evidence package.

## Expected RA Value

The project demonstrates a reproducible post-training research workflow: diagnostic measurement, data curation, controlled baselines, leakage control, and sober reporting. This aligns with RA work in LLM training, data selection, reliable reasoning, and efficient systems.
