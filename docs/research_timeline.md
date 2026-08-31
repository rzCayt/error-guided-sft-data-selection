# Research timeline

This timeline shows how evidence changed the research plan. A stopped branch is retained because it explains why the next experiment was designed differently.

| Stage | Question at the time | What the evidence showed | Decision |
|---|---|---|---|
| 1. Synthetic selector | Can an error profile directly identify useful training examples? | Apparent differences were dominated by coarse metadata; within-group ordering was largely hash-driven. | Stop direct downstream training and test whether the score identifies individual candidates. |
| 2. Candidate utility | Does the score predict one-step utility for individual candidates? | Tulu96 passed its original screening gate; the GSM8K-domain48 boundary check did not. | Treat the signal as limited and domain-sensitive; run a bounded downstream comparison without claiming generality. |
| 3. Original B=500 | Does the initial RDS list beat Random after LoRA training? | Random appeared higher, but sample count did not guarantee matched response-supervision dose or data composition. | Reject the original comparison as a clean policy test and redesign the budget controls. |
| 4. Budget-equivalent 24 cells | Under matched sample count, supervision budget, and source × answer-length composition, does RDS beat Random? | GSM8K difference was +0.480 percentage points with 95% interval [-0.954, +1.889]; OOD macro difference was -0.094 with interval [-1.316, +1.149]. | Report insufficient evidence; do not claim improvement, ineffectiveness, or equivalence. |
| 5. Response-composition audit | Can a frozen training-response feature explain the strict-format behavior difference? | No feature passed both the source-sensitivity and multiplicity gates. | Stop the mechanism branch without GPU retraining. |
| 6. State Dependence v3 | Can candidate utility be measured reliably, and does its ranking transfer across model states? | CPU contracts, an unseen 48-candidate panel, overlap checks, and preflight are complete; no v3 GPU measurement exists. | Test fixed-state reliability first. Cross-state claims remain blocked until that gate passes. |

## Key implementation anchors

- `4517136`: real-model diagnostic stage;
- `c62ac9b`: audited formal Tulu candidate-utility result;
- `5420b2b`: uniform cloud B=500 replication preparation;
- `047d342`: budget-equivalent Phase 1 protocol;
- `6924290`: immutable Phase 2 recovery and audited release base;
- `release/public-research-v0.1`: public narrative, canonical results, and release verification.

Commit identifiers mark code states, not proof by themselves. Current numerical evidence is defined by the hashes in [`results/public_summary/main_results.json`](../results/public_summary/main_results.json).

