# Claim–Evidence Ledger v3

| ID | Claim | Verdict | Evidence | Boundary |
|---|---|---|---|---|
| C1 | Tulu candidate-level error-conditioned score contains incremental local-utility signal under the frozen H1a gate. | Supported for Tulu96 under the original gate | partial Spearman 0.227749; one-sided permutation p=0.072927; positive top-bottom utility | Not reproduced in GSM8K-domain48; does not imply set-level gain |
| C2 | GSM8K-domain candidate utility reproduces the Tulu H1a gate. | Not supported | partial Spearman 0.193935; p=0.185814 | Boundary check only |
| C3 | RDS-error reliably improves GSM8K downstream accuracy over matched Random. | Not supported | +0.480pp; 95% CI [-0.954,+1.889]pp across 24 cells | CI also does not establish equivalence |
| C4 | RDS-error reliably improves OOD arithmetic macro accuracy. | Not supported | -0.094pp; 95% CI [-1.316,+1.149]pp | One model, one budget, one task family |
| C5 | Frozen training-response composition explains the strict-format difference. | Not supported; branch stopped | `CPU_COMPOSITION_NO_GO` | No behavior-constrained training authorized |
| C6 | Candidate utility is reliable at a fixed zero-LoRA state. | Untested; v3 ready for qualification | 144-new-probe U0a contract SHA `8bea3c6a0fbc0d7802279882eaba118a9323f25cfde8f7db97dfc507e90333f4` | Historical values are bridge-only |
| C7 | Candidate utility transfers from zero to final LoRA states. | Untested | v3 U1 contract: 4 states×48 universal-unseen candidates×2 seeds | No state-dependence or stability claim before U0 GO and U1 completion |
| C8 | The v3 primary panel is unaffected by direct training exposure in the four initial adapters. | Supported by CPU audit | 14 exposed candidates removed; 48/48 final candidates unseen; overlap 0; selected-ID SHA `eb8440744cb73ed0582becc6559463bf136fd308823fe95eb286d6adebd3bc23` | Applies only to the four frozen initial states |

## Allowed headline

> A controlled 24-cell LoRA pilot found no reliable downstream advantage for the frozen targeted-selection policy, despite limited candidate-level signal in one domain. A preregistered follow-up now separates fixed-state measurement reliability from cross-state utility transfer.

## Forbidden headlines

- RDS is generally ineffective.
- RDS and Random are equivalent.
- State dependence explains the downstream result.
- Final-adapter local probes reconstruct the optimizer trajectory.
- The current project is already paper-complete.
