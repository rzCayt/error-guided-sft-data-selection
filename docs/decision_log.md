# Decision log

Each entry records the evidence available at the time, the decision, rejected alternatives, and claims that became prohibited.

## 2026-07: stop metadata-dominated synthetic selection

- **Evidence seen:** selector audit showed that coarse task metadata explained apparent score differences and within-group ordering was not candidate-specific.
- **Decision:** do not spend GPU budget on a large downstream matrix; move to candidate-level identifiability.
- **Rejected alternative:** repeatedly redesign the selector after seeing each result.
- **Claim prohibited:** “the synthetic selector identifies useful individual examples.”
- **Code anchor:** early diagnostic line through `4517136`.

## 2026-08: advance Tulu96, retain the failed domain boundary

- **Evidence seen:** Tulu96 partial Spearman 0.227749, one-sided permutation value 0.072927, positive top-minus-bottom utility; GSM8K-domain48 partial Spearman 0.193935 and permutation value 0.185814.
- **Decision:** allow a bounded downstream experiment, but keep the Tulu conclusion domain-limited.
- **Rejected alternative:** discard the GSM8K-domain failure or redefine the screening threshold.
- **Claims prohibited:** “candidate-level signal is domain-general” and “candidate utility implies set-level gain.”
- **Code anchor:** `c62ac9b` and the frozen GSM8K-domain boundary protocol.

## 2026-08: reject the original B=500 comparison as the main policy estimate

- **Evidence seen:** the first nine-cell matrix mixed selection policy with response-supervision dose and data composition.
- **Decision:** preserve the result as history and construct a budget-equivalent common-mix comparison.
- **Rejected alternative:** present the higher Random mean as proof that RDS is harmful.
- **Claim prohibited:** “the original B=500 result isolates the selector effect.”
- **Code anchor:** `5420b2b` followed by the redesigned protocol at `047d342`.

## 2026-08-31: freeze the 24-cell interpretation

- **Evidence seen:** 24 audited cells; GSM8K +0.480 percentage points, 95% interval [-0.954, +1.889]; OOD macro -0.094, interval [-1.316, +1.149].
- **Decision:** report insufficient evidence and retain both beneficial and harmful uncertainty.
- **Rejected alternatives:** “RDS works,” “RDS fails,” and “RDS equals Random.”
- **Claims prohibited:** any general effectiveness or equivalence statement.
- **Code/evidence anchor:** base `6924290`; canonical evidence hashes in `results/public_summary/main_results.json`.

## 2026-08-31: stop the response-composition branch

- **Evidence seen:** no frozen feature passed both prespecified source-sensitivity and multiplicity gates; artifact status `NO_GO`.
- **Decision:** stop without behavior-constrained GPU retraining.
- **Rejected alternative:** search outcome-informed features until one appears significant.
- **Claim prohibited:** “training-response composition explains the observed strict-format difference.”
- **Evidence anchor:** `results/public_summary/evidence/cpu_composition/ARTIFACT_INDEX.json`.

## 2026-08-31: freeze State Dependence v3 as the next experiment

- **Evidence seen:** 14 of 96 historical candidates had direct exposure to at least one target adapter; 82 remained unseen; a balanced 48-candidate panel with zero target-list overlap was frozen.
- **Decision:** remeasure fixed-state utility on all 48 candidates before any cross-state interpretation.
- **Rejected alternatives:** reuse incompatible historical measurements in the primary statistic, or claim state dependence before GPU measurement.
- **Claims prohibited:** “state dependence exists,” “state dependence explains the 24-cell result,” and “final-state probes reconstruct the training path.”
- **Protocol anchor:** `configs/frozen/candidate_utility_state_dependence_protocol_frozen_20260831_v3.json`.
