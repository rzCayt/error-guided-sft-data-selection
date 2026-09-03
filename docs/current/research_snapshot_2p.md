# LLM Post-training Data Selection: Two-page Research Snapshot

Status date: 2026-08-31

## Research question

Under matched sample count, response-supervision tokens, and data composition, does targeted instruction selection outperform matched random selection?

## Controlled study

- Model: Qwen/Qwen2.5-1.5B
- Training: LoRA response-only SFT
- Design: 2 methods x 4 frozen list
  realizations per method x 3 training seeds =
  24 cells
- Budget: 500 examples per list, with matched
  response-supervision exposure and source x answer-length composition
- Primary comparison: rds_error_common_mix - random_common_mix

The RDS lists were generated under distinct query-bootstrap seeds and overlap
substantially. Their effective independence is lower than the nominal list
count; training seeds do not create new selection policies.

## Main results

| Evaluation | RDS minus Random | 95% interval | Judgment |
|---|---:|---:|---|
| GSM8K exact numeric accuracy | +0.480 pp | [-0.954, +1.889] pp | Insufficient evidence |
| Three-task OOD macro accuracy | -0.094 pp | [-1.316, +1.149] pp | Insufficient evidence |

The controlled block found no reliable downstream accuracy advantage for the
frozen error-conditioned selection policy. The intervals still include practically relevant positive
and negative effects, so the study does not establish ineffectiveness or
equivalence.

## Candidate-to-set gap

- Tulu96 partial Spearman: 0.228;
  one-sided permutation p = 0.073;
  met the preregistered screen: partial Spearman >= 0.15, one-sided p <= 0.10,
  and positive top-minus-bottom utility.
- GSM8K-domain48 partial Spearman:
  0.194; one-sided permutation
  p = 0.186; did not meet
  the same screen because p exceeded 0.10.
- The limited candidate-level signal did not become a stable downstream gain
  when 500 selected examples were trained together.

## Fixed-state reliability follow-up

The next study first measures fixed-state candidate-utility reliability:

- frozen panel: 48 candidates;
- direct training overlap: 0;
- 48 candidates × 3 repeated fixed-state measurements;
- status: CPU preflight complete; GPU qualification and formal measurement
  have not started.

Decision rule:

1. unreliable fixed-state measurement -> study measurement uncertainty;
2. reliable measurement with changing rankings -> study candidate revaluation;
3. reliable and stable rankings -> study redundancy, conflict, and
   complementarity within training sets.

## Four-week research module

1. qualify the fresh-process utility runner and seed semantics;
2. complete and audit the 48 × 3 fixed-state reliability panel;
3. apply the predefined stop/go decision before any cross-state evaluation;
4. deliver code, evidence manifests, a bounded result memo, and the next
   falsifiable experiment.

## Claim boundaries

Supported: no reliable RDS advantage was observed in this setting; Tulu96
met the preregistered candidate screen while GSM8K-domain48 did not.

Not supported: RDS is generally ineffective; RDS and Random are equivalent;
state dependence has been observed; a local final-adapter probe reconstructs
the historical optimizer trajectory.
