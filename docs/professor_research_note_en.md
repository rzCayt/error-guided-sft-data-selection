# When Is Error-Guided Data Selection Identifiable?

**Ruizhe Cao · Research note · July 2026**

## Question

Can diagnostic failures from a small language model identify training examples
that are more useful than examples selected by strong matched controls? This is
a data-centric post-training question: before paying for an SFT sweep, the
selection score should first contain candidate-level information that is not
already encoded by task metadata.

## Method

I designed the research question and audited an AI-assisted, test-driven
pipeline containing synthetic task generation, split checks, raw-output
retention, parser audits, matched baselines, permutation controls and frozen
decision gates. Model-native prompts are separated from scoring rules. The test
split is not used for selector development.

Three increasingly specific selectors were examined:

1. A metadata error-profile selector matched on task family, difficulty,
   answer magnitude and reasoning length.
2. A residual selector using operation structure after the same coarse controls.
3. A tiny model-aware pilot that compares candidate gradients/loss responses to
   error and correct query groups.

## Evidence

The metadata audit reproduced exactly across runs (500 candidates, budget 128):
all non-random score fields were already included in the matching key, so within
a stratum the ranking reduced to a hash tie-breaker. The residual selector also
failed: scores were constant after full operation signature was fixed. The
model-aware pilot created within-stratum variation, but its observed effect
size, 0.0193, did not exceed the preregistered permutation 90th percentile,
0.0241. A separate 25-item Qwen3-1.7B pipeline check reached 19/25 numeric
accuracy and 25/25 parse success; this validates the interface and parser chain,
not the selector.

## Conclusion and next experiment

The present selector is not identifiable beyond matching variables, and there
is no evidence yet that it predicts training utility. The correct next step is
H1a: preregister a candidate-utility proxy, evaluate whether it predicts
held-out gradient or loss reduction across independent seeds, and proceed to a
small LoRA comparison only if that gate passes. A negative H1a result would
still be useful because it would rule out an expensive training matrix and
clarify which selection signals are merely metadata proxies.

## Contribution and limitations

My contribution is the problem formulation, evidence audit, claim boundaries
and decision-gated experimental design; coding was AI-assisted and every public
claim is tied to stored artifacts and tests. The study is development-only,
synthetic and small-scale. It establishes neither generalization nor training
effectiveness.
