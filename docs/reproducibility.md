# Reproducibility guide

## Reproduction levels

### Level 1: unit and integrity checks (CPU)

```powershell
python -m pip install -e ".[dev]"
pytest -q
python scripts/validate_public_release.py
```

This checks deterministic data generation, split controls, parser behavior,
selector audit logic, invariance checks, and result-building utilities.

### Level 2: offline selector audits (CPU)

Write reruns to new directories so frozen evidence is not overwritten:

```powershell
python scripts/audit_selector_identifiability.py `
  --output-dir results/reproduction/selector_identifiability

python scripts/audit_residual_selector_identifiability.py `
  --output-dir results/reproduction/residual_selector_identifiability
```

Expected decisions:

- metadata selector: `not_identifiable_beyond_matching_metadata`;
- residual selector: `not_identifiable_beyond_static_operation_metadata`.

### Level 3: model-aware feasibility (CUDA)

The frozen model-aware run used:

- model: `Qwen/Qwen3-1.7B`;
- revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
- PyTorch: `2.8.0+cu128`;
- Transformers: `4.57.2`;
- dtype: `bfloat16`;
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB class).

Run F0/F1 before F2 and use new output directories:

```powershell
python scripts/run_model_aware_f0_f1.py `
  --output-dir results/reproduction/model_aware_f0_f1

# F2 is frozen to its preregistered inputs and should only be rerun after F0/F1.
python scripts/run_model_aware_f2.py `
  --output-dir results/reproduction/model_aware_f2
```

The F2 result is expected to remain a tiny-pilot negative result. A successful
program exit is not a successful research claim.

## Data controls

- Selector development uses development diagnostics only.
- Test-split information is not used to construct the selector.
- Candidate answers and rationales are excluded from the residual-selector
  score where specified by the audit metadata.
- Generated outputs and parser modes are stored for inspection.

## Reproduction limits

- CUDA kernels and package versions can introduce small numerical differences.
- Model weights are referenced by repository revision and are not stored here.
- The 25-item model check is bounded pipeline validation, not a statistical
  estimate of general performance.
- The eight-candidate F2 run is too small for a general utility claim.
- `scripts/build_public_release_artifacts.py` is a maintainer utility: its
  default source is the local provenance directory, which is intentionally not
  part of the public release because it contains machine-specific cache paths.
  The sanitizer itself is tested with a self-contained synthetic fixture.
