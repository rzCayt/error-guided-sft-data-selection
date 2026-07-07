# Error-Guided SFT Data Selection

**Research question.** Can base-model diagnostic errors guide SFT data selection more effectively than matched random selection under the same sample and token budget?

This repository is a small, reproducible research scaffold for RA applications around post-training, data-efficient SFT, reasoning failure analysis, and data selection. Finance-style variables are used only as a familiar shell for solver-verifiable numerical reasoning. The project is not a finance LLM, investment advice system, or domain QA benchmark.

## Abstract

Small language models often fail on numerical reasoning because they bind variables incorrectly, apply the wrong formula, or lose scale across multi-step prompts. This project studies whether those base-model diagnostic failures can be converted into a data-selection signal for efficient LoRA supervised fine-tuning. A deterministic generator creates four families of solver-verifiable numerical problems: ratio change, multiplicative relation, weighted aggregation, and temporal numeric constraints. The pipeline first runs a locked diagnostic set through a base model, maps failures into an error taxonomy, and then selects training examples from a separate candidate pool whose metadata matches the observed weaknesses. A matched random baseline controls for task family, difficulty, answer magnitude, and reasoning length under the same sample budget. Locked in-distribution and out-of-template tests are reserved for evaluation only. The intended contribution is a clear, auditable comparison of targeted error-guided SFT data against carefully matched random selection, with emphasis on leakage control and reproducible evidence rather than inflated performance claims.

## Current Scope

- Deterministic synthetic data generator and solver.
- Locked split names: `candidate_pool`, `dev_diagnostic`, `test_id`, `test_ood_template`, `test_ood_range`.
- Error taxonomy and simulated base-diagnostic script for no-GPU/no-model environments.
- Error-guided and matched-random selectors.
- LoRA smoke interface with a no-training evidence fallback.

## Quick Start

```powershell
cd E:\RA准备\07_error_guided_sft_repo
python -m pip install -e .[dev]
python scripts/generate_data.py --all
python scripts/run_base_diagnostic.py
python scripts/build_selection_sets.py --budget 128
python scripts/evaluate_results.py
pytest
```

Optional training dependencies can be installed with:

```powershell
python -m pip install -e .[train]
python scripts/run_lora_smoke.py --model Qwen/Qwen2.5-0.5B
```

If local GPU or training packages are unavailable, `run_lora_smoke.py` writes a no-training evidence package instead of blocking the project.

## Repository Map

- `docs/project_spec.md`: research design, split discipline, evaluation plan.
- `docs/data_generation_spec.md`: task families, schema, deterministic solver guarantees.
- `docs/selection_policy_spec.md`: targeted and matched-random policies.
- `src/eg_sft/data`: generator, schemas, solver.
- `src/eg_sft/eval`: parser, metrics, error taxonomy.
- `src/eg_sft/selection`: selection policies and bias audit.
- `scripts`: reproducible command-line entry points.

## Quality Rules

- Do not use test results to adjust selection.
- Do not mix `dev_diagnostic` and test splits.
- Do not commit model weights, adapters, API tokens, or large generated files.
- Report failed training attempts as evidence with environment details and next steps.
