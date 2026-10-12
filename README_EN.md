# LLM Post-training Data Selection: Controlled Evidence and Reliable Measurement

[中文主页](README.md) · [Claims and limitations](CLAIMS_AND_LIMITATIONS.md) · [Reproduction](REPRODUCE.md)

## Research question

Does targeted data selection improve post-training more reliably than matched random selection when sample count, response-supervision tokens and data composition are controlled?

The completed study uses Qwen2.5-1.5B Base, response-only LoRA supervised fine-tuning (SFT), one frozen RDS targeted policy and matched Random. LoRA trains a small set of added parameters; response-supervision tokens are the answer tokens contributing to the loss.

## Completed evidence

**24 cells** = 2 methods × 4 list realizations × 3 training seeds, with 500 examples per list. Supervision budgets and source × answer-length composition are matched. Values come from the [canonical summary](results/public_summary/main_results.json).

| Metric | RDS minus Random | 95% interval |
|---|---:|---:|
| GSM8K numeric accuracy | +0.480 percentage points | [-0.954, +1.889] |
| Three-task out-of-domain mean accuracy | -0.094 percentage points | [-1.316, +1.149] |

No reliable advantage was observed in this setting. The intervals admit positive and negative effects, so they do not establish no effect or equivalence. The targeted lists overlap substantially; their effective independence is below the nominal four, and training seeds do not create new selection policies. The out-of-domain tasks are SVAMP, ASDiv numeric and MultiArith.

A limited candidate-level screening signal did not translate into a stable set-level gain. A response-composition mechanism audit failed its preset gate and was stopped without more retraining. Frozen evidence and negative results are preserved.

## Current exploration

Three candidate explanations are considered in order: measurement noise, changes in value rankings after training, and redundancy/conflict/complementarity in sets. Candidate utility means the decrease in independent validation loss after one standardized candidate update.

The first step reuses the frozen 48-candidate × 3-probe-seed reliability protocol. **GPU qualification and formal measurements have not started.** Cross-state work depends on its original gate; preparation is not a research result. New studies have a [separate registry and roadmap](docs/current/research_upgrade_20261013.md).

An SFT-to-GRPO engineering path is also available. GRPO samples several responses per prompt and trains using their relative rewards. CPU tiny-model integration tests do not establish CUDA feasibility, mathematical gains or novelty.

## Quick verification

```bash
python -m pip install -e ".[dev]"
python scripts/reproduce_public_summary.py --check
python scripts/verify_research_upgrade.py
python scripts/verify_public_release.py
python -m ruff check .
```

These verify permitted public summaries and provenance. Tests, GPU reproduction and AutoDL commands are documented in the reproduction guide. Raw generations, adapter weights, restricted dataset text and credentials are not committed.

## Ownership and development

The owner framed questions, authorized freezes and runs, inspected evidence and accepted or stopped branches. AI contributed substantially to implementation, debugging and documentation; see [AI assistance](AI_ASSISTANCE.md). Owner mastery requires personal explanations, modifications and failure diagnosis, not merely an AI-written repository.

The internal experiment-integrity audit remains WARN: large raw generations and adapters are outside Git, and an independent audit of the underlying experiments has not passed. This upgrade adds engineering preparation; it does not change historical claim boundaries.
