# Current research overview

## Question

Under matched sample count, response-supervision tokens, and source × answer-length composition, does the frozen RDS targeted-selection policy outperform matched Random for LoRA post-training?

## Completed evidence

- Qwen2.5-1.5B Base with response-only LoRA SFT;
- 2 methods × 4 list constructions × 3 training seeds = 24 audited cells;
- 500 examples per list under the frozen supervision and composition constraints;
- GSM8K plus SVAMP, ASDiv numeric, and MultiArith evaluation;
- candidate-level Tulu96 and GSM8K-domain48 measurements;
- frozen CPU response-composition mechanism audit.

## Result

RDS minus Random was +0.480 percentage points on GSM8K with 95% interval [-0.954, +1.889], and -0.094 percentage points on the three-task OOD macro with interval [-1.316, +1.149]. The evidence is insufficient for improvement, harm, or equivalence claims.

## Next experiment

State Dependence v3 first tests repeated candidate-utility measurement at one fixed zero-LoRA state. Cross-state measurement begins only if the fixed-state reliability gate passes. CPU contracts and preflight are complete; no v3 GPU result exists.

See the [claim–evidence ledger](claim_evidence_ledger.md), [timeline](../research_timeline.md), and [canonical result JSON](../../results/public_summary/main_results.json).
