# Data Generation Spec

## Design Goal

Create solver-verifiable numerical reasoning tasks with controlled metadata. The wording can use commerce or finance-style variables, but every answer is deterministic and independent of external financial knowledge.

## Task Families

### Ratio Change

Given an initial value and percentage increase/decrease, compute the final value or net change.

Example: "A metric starts at 120 and increases by 15%. What is the final value?"

### Multiplicative Relation

Given chained multipliers or unit counts, compute a derived value.

Example: "Each unit contains 4 packs and each pack has 12 items. How many items are in 7 units?"

### Weighted Aggregation

Compute a weighted average or total from two to four components.

Example: "Group A has weight 0.35 and value 80, Group B has weight 0.65 and value 92. What is the weighted value?"

### Temporal Numeric Constraint

Track a value over ordered periods with additions, removals, caps, or minimum constraints.

Example: "A count is 50 on Monday, gains 8 on Tuesday, then loses 6 on Wednesday. What is the Wednesday value?"

## Schema

Each example contains:

- `id`
- `split`
- `task_family`
- `difficulty`
- `prompt`
- `answer`
- `rationale`
- `metadata`
- `buckets`

Required buckets:

- `difficulty_bucket`
- `answer_magnitude_bucket`
- `reasoning_length_bucket`

## Determinism

Every script accepts a seed and writes reproducible JSONL. The generator uses only Python standard-library random state passed through `random.Random`.

## Leakage Rules

- Prompts are generated independently by split seed.
- Selection can read candidate metadata and dev diagnostic error profile.
- Selection must not inspect test predictions or evaluation metrics.
- OOD template examples use held-out surface templates.
- OOD range examples use larger numeric ranges than the candidate and dev splits.
