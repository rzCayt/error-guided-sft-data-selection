# Selection Policy Spec

## Error-Guided Policy

Input:

- `data/samples/candidate_pool.jsonl`
- `results/error_profile_v0.csv`

The selector scores candidate examples by matching the diagnostic weakness profile:

1. Higher weight for task families with lower diagnostic accuracy.
2. Higher weight for difficulty buckets with more failures.
3. Higher weight for answer-magnitude and reasoning-length buckets with more failures.
4. A per-stratum cap that normally prevents the targeted subset from taking more than half of any matching stratum, preserving enough same-stratum examples for matched random.
5. Tie-break with deterministic stable hashing from `id` and seed.

The selector does not use test labels, test predictions, or test metrics.

## Matched Random Policy

Matched random must preserve, as closely as possible, the targeted subset distribution over:

- task family
- difficulty bucket
- answer magnitude bucket
- reasoning length bucket

Within each stratum, examples are sampled with a deterministic random seed. The targeted selector is capped by stratum so the matched-random selector can usually avoid overlap while preserving exact matching. If a stratum still does not have enough non-targeted examples, matched random allows overlap with the targeted subset rather than breaking stratum matching. This makes the baseline distribution fairer, but the overlap count must be reported because it can make the two training sets less independent.

## Bias Audit

The audit reports distribution differences between targeted and matched-random selections:

- count by matching stratum
- count by task family
- count by difficulty
- mean answer magnitude
- mean reasoning length
- targeted/matched overlap rate

Large deviations are a warning that downstream comparisons may reflect distribution shift rather than better data selection.
