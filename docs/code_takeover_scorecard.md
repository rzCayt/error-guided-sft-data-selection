# Personal code-takeover scorecard

This scorecard is intentionally not an automated certificate. Fill it in after
doing the work yourself; evidence means a file, line, row, command output, or
recording timestamp.

## A. Parser-from-scratch exercise (20 points)

Create a file outside the repository's implementation, for example
`takeover_work/parser_from_scratch.py`. Do not copy the existing parser.

Required behavior:

- accept a final line in the form `Final answer: <number>`;
- support negatives and decimals;
- reject a missing marker;
- reject a marker whose payload is a formula or placeholder;
- state and test what happens when multiple markers appear.

Scoring:

| Item | Points | Your evidence |
|---|---:|---|
| Parser written from an empty file | 5 | |
| At least five tests, including malformed cases | 5 | |
| Behavior agrees with the stated specification | 5 | |
| Can explain one design trade-off without notes | 5 | |

## B. One-row evidence trace (20 points)

Choose one correct and one incorrect row from the public 25-item JSONL.

| Item | Points | Your evidence |
|---|---:|---|
| Gold answer recomputed independently | 5 | |
| Parser output reproduced by hand | 5 | |
| Summary count reconciled | 5 | |
| Claim boundary explained | 5 | |

## C. Selector failure explanation (20 points)

| Item | Points | Your evidence |
|---|---:|---|
| Explains exact matching and strata | 5 | |
| Identifies the original hash tie-breaker | 5 | |
| Explains residual operation-signature constancy | 5 | |
| Distinguishes stability from effectiveness | 5 | |

## D. Experimental design and statistics (20 points)

| Item | Points | Your evidence |
|---|---:|---|
| Explains permutation versus bootstrap | 5 | |
| Explains the F2 effect-size gate | 5 | |
| Designs H1a without test-split access | 5 | |
| States an escalation and a stop rule | 5 | |

## E. Fifteen-minute research presentation (20 points)

Suggested timing:

- 2 minutes: problem and why selection identifiability comes before SFT;
- 3 minutes: data, splits, model interface, and parser;
- 4 minutes: metadata and residual-selector failures;
- 3 minutes: model-aware F0/F1/F2 and failed gate;
- 2 minutes: H1a next experiment;
- 1 minute: limitations and personal contribution.

| Item | Points | Your evidence |
|---|---:|---|
| Finishes in 13–16 minutes | 5 | |
| Every number has a named artifact | 5 | |
| Clearly separates completed and future work | 5 | |
| Answers three unscripted questions | 5 | |

## Passing rule

- Minimum total: 80/100.
- No section below 12/20.
- Automatic fail if you claim completed SFT effectiveness, use the test split to
  design the selector, or cannot trace the 19/25 count to raw rows.

After passing, record the date and the two weakest topics to revisit. Until
then, the accurate description is “AI-assisted pipeline that I designed,
audited, and am taking over,” not “independently implemented end-to-end.”
