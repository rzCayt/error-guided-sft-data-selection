# Current result index

This page lists the public current-stage artifacts. Historical result files are intentionally excluded.

| Artifact | Purpose | Identity |
|---|---|---|
| [`main_results.json`](../../results/public_summary/main_results.json) | Canonical machine-readable results and claim boundaries | Recomputed by `scripts/reproduce_public_summary.py` |
| [`main_results.csv`](../../results/public_summary/main_results.csv) | Flat downstream result table | Generated from the canonical JSON |
| [`main_results_table.md`](../../results/public_summary/main_results_table.md) | Human-readable generated table | Generated from the canonical JSON |
| [`experiment_registry.csv`](../../results/public_summary/experiment_registry.csv) | 24 cell IDs, method/list/train seeds, and selection hashes | Generated from the frozen canonical matrix |
| [`figures/manifest.json`](../../figures/manifest.json) | Figure source and output hashes | Source SHA must equal the canonical JSON SHA |
| [`configs/frozen/MANIFEST.json`](../../configs/frozen/MANIFEST.json) | Byte-identical public config snapshots | Each snapshot must match its runtime source |
| [`claim_evidence_ledger.md`](claim_evidence_ledger.md) | Supported, unsupported, and untested claims | Every numeric claim points to a public evidence artifact |

The evidence file hashes used to construct the canonical result are stored inside `main_results.json`. Run `python scripts/verify_public_release.py` to check every identity.
