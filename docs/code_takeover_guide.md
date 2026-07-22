# Code takeover guide

The goal is not to memorize the repository. The goal is to trace one claim from
input to raw output, parser, metric and decision gate without assistance.

## Seven modules to own

1. **Generator** — explain task families, answer construction and why synthetic
   wording can become a confound.
2. **Solver/gold path** — show where the numerical target is calculated and how
   it is separated from model output.
3. **Splits** — explain deterministic IDs, overlap audits and why the test split
   is unavailable during selector design.
4. **Inference interface** — show model revision, chat template, decoding and
   raw-output retention.
5. **Parser** — trace `raw_continuation` to `parsed_prediction`, parser mode and
   numeric equality.
6. **Selector/baseline** — explain matching fields, within-stratum ranking and
   why a hash tie-breaker is not candidate-level signal.
7. **Statistics/gates** — distinguish permutation controls, bootstrap stability,
   paired seeds and a preregistered threshold.

## Mandatory practical checks before emailing

- [ ] Draw the data flow from generator to summary table from memory.
- [ ] Select one raw output and reproduce its parser result by hand.
- [ ] Explain why both metadata and residual selectors failed.
- [ ] Explain permutation, bootstrap and paired-seed designs without notes.
- [ ] From an empty file, implement a minimal `Final answer:` numeric parser and
      five tests, including negatives, decimals and malformed outputs.
- [ ] Reproduce the 25-item model check and reconcile every row count.
- [ ] Record a 15-minute presentation and answer the questions below.

## Technical interview questions

1. What causal claim would a targeted-vs-random SFT comparison support, and what
   confounds must be controlled first?
2. Why does exact matching make the original selector non-identifiable?
3. Why is a different selected ID set not sufficient evidence of a different
   signal?
4. What does the model-aware pilot's failed effect-size gate mean?
5. Why can parser improvements change apparent model accuracy without changing
   model capability?
6. Why are the 25-item and 100-item model scores not directly comparable?
7. How would you design H1a without accessing the final test split?
8. What evidence would justify spending compute on LoRA?
9. Which parts were AI-assisted, and how did you verify them?
10. Name one result that would falsify your preferred hypothesis.

## Passing standard

Pass only when every answer links to a concrete file or row and no answer uses
“the agent did it” as a substitute for understanding. Until then, describe the
work as an AI-assisted research pipeline that you designed and audited, not as
an independently implemented post-training system.
