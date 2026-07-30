# Public Evidence Package Re-Audit

**Date**: 2026-07-31
**Auditor**: Independent read-only audit agent
**Scope**: `rds_full_pool_10k_public_evidence_v1` plus the two original B=500 selection manifests

## Overall Verdict: PASS

The public evidence package and the two original manifests intended for the matrix pass this independent re-audit. The two absolute-path findings in the local-only v1 evidence are explicitly excluded from the public package, while the public contract cryptographically binds back to the unchanged local run contract.

## Results

| Check | Status | Evidence |
|---|---|---|
| Public-manifest self-hash | PASS | `manifest_content_sha256` recomputes to `19f9b406e2268b0124017126b636f70e0872178956122249ff051526d43ac2ee`. |
| Listed file integrity | PASS | All 85 listed files exist, have the listed byte count and SHA-256, and no extra package file exists. |
| Public selection byte identity | PASS | Package `rds_all` and `rds_error` files each have exactly the same SHA-256 as their original manifest. |
| Path/privacy scan | PASS | Full UTF-8 scan: zero Windows/Unix private-path, username, secret-like, or raw-source-text-field hits. |
| Sanitization/binding | PASS | No local `run_contract.json`, legacy `data_manifest.json`, or `.pt` is in package. `public_run_contract.json` changes only `command[0]` to `python` and binds the original local contract file/self hashes. |
| Local evidence immutability | PASS | Original run-contract, inventories, score file, metrics, and finalization hashes remain identical to the v1 audit record. |
| Public recomputation | PASS | 10,000 candidates; 8,542 eligible; 1,458 excluded; 448 queries; 99 error queries; both rank permutations and both B=500 top-500s reproduce exactly. |

## Exact Hashes and Recomputed Metrics

- Public manifest file SHA-256: `c5c409aaab0121eba0f3b5f62e2658200b32038730ce9fa02cbe9e5d2f355830`
- Public manifest content SHA-256: `19f9b406e2268b0124017126b636f70e0872178956122249ff051526d43ac2ee`
- Public contract file SHA-256: `77c7fba539db10038e5e7f8e6f94712a8564860b8cc80b1c082c795771b8772c`
- Original local run-contract file/self SHA-256: `006b4e50c954ab1c1b25bbb5bc7ee4b2d4570e791c7aa7832bf51400013eddff` / `175add6c3083f7d1e697819a8db1fbeee57d91b33e6907b313a38749f9cfeb5d`
- `rds_all` original/public manifest SHA-256: `891ca5b51840f3a914659c825b698222ee5943e5f4b0d3433635fa7813098f65`
- `rds_error` original/public manifest SHA-256: `ac050c659e09dfd0fbd40608ab2fa092ebfba90bff0a838a79053363cbaeff13`
- Candidate / eligible / query order SHA-256: `dba131171a3f434ef1f1f7a2fdb459a5da3dffbaab8f50687c8ef06cf5120beb` / `7006e65ba427644bc2b538f3a313dcb161cb0e97aae47657719b2fa55b639335` / `45db76acb8edd5d641b9b227e56ac87011e110a82155c362d1ab207991209d6d`
- Recomputed rank Spearman: `0.9975506350397658`
- Recomputed top-500 intersection / Jaccard: `458` / `0.8450184501845018`
- `rds_all` selected-ID SHA-256: `e043829d1a5cf8c6eb65253632ac6a7d151082eb08d252c5fc96a172cf53b7b0`
- `rds_error` selected-ID SHA-256: `6442105cec94c6b2f461e534a0df177fbb4edc03d733f9bd87de8770a65bf4d8`

The complete machine-readable re-audit trace is embedded in `EXPERIMENT_REAUDIT.json`. No original evidence, code, or configuration was changed.

## Limitation

The external reviewer MCP prescribed by the general audit skill was unavailable. This report is an independent deterministic re-audit by a separate read-only agent, not a cross-model reviewer verdict.
