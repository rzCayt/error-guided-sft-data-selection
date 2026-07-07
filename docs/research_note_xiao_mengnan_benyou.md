# Research Note: Xiao Li / Mengnan Du / Benyou Wang-Oriented Version

## Angle

The project studies trustworthy and efficient reasoning improvement for small language models. Instead of adding more data indiscriminately, it asks whether failure-aware data selection can improve numerical reasoning under a strict budget.

## Technical Core

- Solver-verifiable numerical tasks.
- Locked diagnostic and test splits to reduce leakage.
- Error taxonomy for arithmetic, formula, scale, temporal ordering, parse, and variable-binding failures.
- Matched random baseline controlling for task family, difficulty, answer scale, and reasoning length.
- LoRA SFT smoke path for small open models.

## Why It May Fit

The project is compact enough for a new RA to execute, but it touches several durable research themes: trustworthy evaluation, data-efficient training, reasoning failure analysis, and robust comparison design.

## Next Evidence

Run real base diagnostics and B128/B256 LoRA comparisons on GPU, then report per-error and per-family improvements without tuning on the test sets.
