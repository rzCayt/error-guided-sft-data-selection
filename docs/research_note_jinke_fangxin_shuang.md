# Research Notes: Jinke Ren / Fangxin Wang / Shuang Li

## Jinke Ren-Oriented Version

**Angle:** agentic data curation for post-training.

The project can be presented as a first step toward an agentic data-curation loop: diagnose base-model failures, summarize weakness profiles, select candidate training data, audit distribution shift, and rerun evaluation under locked budgets. The current implementation is deterministic and auditable; a later version could add an LLM agent that proposes new candidate templates while a symbolic solver verifies labels.

## Fangxin Wang-Oriented Version

**Angle:** efficient LLM systems and routing.

The repository studies whether small models can receive more value per training token when the training subset is routed toward observed failure modes. This connects to efficient model adaptation, budget-aware data routing, and systems-style tradeoffs between diagnostic cost, SFT cost, and evaluation gain.

## Shuang Li-Oriented Version

**Angle:** reliable reasoning and failure analysis.

The main reliability contribution is not a new benchmark alone, but a leakage-controlled workflow for converting reasoning failures into a testable intervention. The task labels are solver-verifiable, the error taxonomy is explicit, and matched random controls make it harder to attribute gains to accidental task-distribution differences.
