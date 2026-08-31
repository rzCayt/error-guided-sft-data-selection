# Claims and limitations

## Supported by completed evidence

1. The audited 24-cell block did not find a reliable downstream accuracy advantage for the frozen RDS policy over matched Random in the tested setting.
2. Tulu96 passed its original candidate-utility screening gate; the 48-candidate GSM8K-domain boundary check did not.
3. The frozen response-composition mechanism did not pass its prespecified CPU gate, so that branch stopped without GPU retraining.
4. The frozen State Dependence v3 panel contains 48 candidates with zero direct training overlap across the four initial target adapters.

## Not supported

The evidence does not establish that:

- RDS is generally ineffective;
- RDS and Random are equivalent;
- the error-conditioned policy adds information beyond the all-query policy;
- state dependence exists or explains the downstream result;
- final-adapter local probes reconstruct the historical optimizer trajectory;
- the conclusions transfer to other model scales, budgets, domains, or selectors.

## Main limitations

- one base model and scale: Qwen2.5-1.5B Base;
- one main training budget: 500 examples under the frozen response-supervision budget;
- one candidate pool and arithmetic-focused evaluation;
- only four targeted-list constructions, with high overlap among those lists;
- training-seed replication does not create additional independent selection policies;
- candidate-level utility and set-level downstream effects answer different questions;
- State Dependence v3 has completed CPU preflight only and has no GPU result.

## Statistical interpretation

“Insufficient evidence” means that the uncertainty interval contains both beneficial and harmful effects. It does not mean “no effect,” and it does not establish practical equivalence. The canonical point estimates and intervals are stored in [`results/public_summary/main_results.json`](results/public_summary/main_results.json).

