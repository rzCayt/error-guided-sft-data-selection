# Code and evidence map

| Research responsibility | Primary implementation | Verification or evidence |
|---|---|---|
| Frozen data preparation and split checks | `src/eg_sft/` and `scripts/prepare_public_gsm8k_v1.py` | data manifests, hashes, split and leakage tests |
| Strict numeric answer parsing | parser modules under `src/eg_sft/` | parser tests and raw-output recomputation audits |
| Candidate scoring and utility measurement | scoring/utility modules under `src/eg_sft/`; H1a scripts under `scripts/` | `results/public_summary/evidence/h1a/` |
| Budget-equivalent list construction | Phase 2 selection modules and frozen configs | 500-example, token-budget, composition, determinism, and list-SHA tests |
| LoRA training and immutable recovery | Phase 2 runner modules under `src/eg_sft/experiment/` and `scripts/` | training logs, adapter reload proof, recovery tests, per-cell audits |
| GSM8K and OOD evaluation | evaluation workers under `scripts/` | raw outputs and formal/OOD audit artifacts |
| Public result aggregation | `scripts/reproduce_public_summary.py` | `results/public_summary/main_results.json`, CSV, and Markdown table |
| State Dependence v3 panel and preflight | `scripts/freeze_state_dependence_panel_v3.py`, `scripts/preflight_candidate_utility_state_dependence_v3.py` | `artifacts/state_dependence_*_v3.json` |
| State Dependence v3 execution and analysis | `scripts/run_candidate_utility_state_drift.py`, `scripts/analyze_state_dependence_u0_v3.py`, `scripts/analyze_state_dependence_u1_v3.py` | CPU contracts and tests only; no GPU result yet |
| Public release integrity | `scripts/verify_public_release.py` and GitHub Actions | hashes, claims, links, secret/path scan, full CPU tests |

The table names stable responsibilities rather than every historical script. Earlier implementations remain under [`docs/history/`](history/) and release snapshots so that current entry points are not confused with superseded ones.
