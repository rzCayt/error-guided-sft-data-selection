# Public-release code identity repair

Date: 2026-08-31

## Problem found

The `6924290` code state included later fixes to training recovery and historical-drift diagnostics, but the active Phase 2 semantic-code manifest still contained hashes generated before those fixes. The full CPU suite exposed the inconsistency: using the newer code failed the frozen-manifest authority test, while restoring the older code failed the newer behavior tests.

## Resolution

The public release keeps the latest `6924290` behavior and regenerates the active semantic-code and canonical-runtime manifests with the repository's official builders. The original manifests are retained in `releases/phase2_v8/`.

| Identity | Original release SHA-256 | Public repaired SHA-256 |
|---|---|---|
| Semantic code manifest | `3f759906adabfcba5388187dc72691c8009fa600d1df759f606c4e7a1345624e` | `13efb94e4fa29699e554a76f29725ad3b1a97c37b521363cb7b39db0293ae049` |
| Canonical runtime manifest | `e68eddfac5c8bcc6225d67066dc1c267b8ba767c1db5173047107d5729418fa0` | `6d55e1cbbc9b753d25dc9498296c0100967224ae502d9062944e77bc4a20bd93` |

## Boundary

This repair changes only the identity record for the current public code. It does not alter selected examples, model outputs, metrics, confidence intervals, or any completed research claim. Historical deployment identities remain available in the release snapshot.

