# Selection Policy Spec

## Error-Guided Policy

Input:

- `data/samples/candidate_pool.jsonl`
- `results/error_profile_v0.csv`

The selector scores candidate examples by matching the diagnostic weakness profile:

1. Higher weight for task families with lower diagnostic accuracy.
2. Higher weight for difficulty buckets with more failures.
3. Higher weight for answer-magnitude and reasoning-length buckets with more failures.
4. Tie-break with deterministic stable hashing from `id` and seed.

The selector does not use test labels, test predictions, or test metrics.

## Matched Random Policy

Matched random must preserve, as closely as possible, the targeted subset distribution over:

- task family
- difficulty bucket
- answer magnitude bucket
- reasoning length bucket

Within each stratum, examples are sampled with a deterministic random seed. If a stratum does not have enough unused examples, the deficit is filled from the nearest remaining pool while retaining a full audit row.

## Bias Audit

The audit reports distribution differences between targeted and matched-random selections:

- count by matching stratum
- count by task family
- count by difficulty
- mean answer magnitude
- mean reasoning length

Large deviations are a warning that downstream comparisons may reflect distribution shift rather than better data selection.
