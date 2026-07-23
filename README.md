# Error-Guided SFT Data Selection

This project asks a narrow question: can diagnostic failures from a target
language model identify training examples that are unusually useful for
post-training?

## Status: candidate-utility gate passed; downstream SFT untested

The original synthetic selector audits remain negative. A newer public-data
extension now provides a preregistered reason to run the bounded SFT comparison.

- The first selector is not identifiable beyond the metadata already used by
  an exact matched-random baseline.
- A residual operation-aware selector remains constant after static operation
  metadata is fixed.
- An eight-candidate model-aware feasibility pilot fails one preregistered
  effect-size gate (`0.0193 < permutation p90 0.0241`).
- In a frozen 96-candidate Tulu experiment, the error-conditioned RDS+ score
  predicts one-step candidate utility after controlling the all-query score
  and training-token length: partial Spearman `0.22775`, one-sided
  1,000-label-permutation `p=0.07293`, and positive top-minus-bottom utility.
- No completed result yet shows that targeted selection improves downstream
  LoRA/SFT accuracy or beats all-query RDS+ or random selection after training.

The candidate-utility gate is therefore passed for the frozen Tulu pool. This
authorizes the preregistered bounded `B=500` comparison; it is not itself an SFT
effectiveness result.

## Verified evidence

| Check | Scope | Result | Interpretation |
|---|---:|---:|---|
| Original selector identifiability | 500 candidates, budget 128 | Fail | Score is fully controlled by matched metadata |
| Residual selector identifiability | 500 candidates, budget 128 | Fail | Score is still static at the operation-signature level |
| Model-aware feasibility | 8 candidates | Fail | Representation is computable, but the effect gate is not cleared |
| Formal Tulu candidate utility H1a | 96 candidates, 1,000 permutations | Pass | Incremental candidate-level signal clears all three preregistered gates |
| Selector reproduction | 500 candidates, budget 128 | Exact SHA-256 match | Offline audit is deterministically reproducible |
| Model pipeline check | Qwen3-1.7B, 25 mixed-family dev items | 19/25 numeric; 25/25 parsed | Raw output-to-parser-to-metric chain works end to end |

The 19/25 pipeline check uses the first 25 items of the mixed-family
`data/samples/dev_diagnostic.jsonl`. It must not be compared directly with the
older 8/25 Qwen3-1.7B row in the model-native baseline table, which uses a
different weighted-aggregation-only input set.

## Read this first

1. [Research note](docs/professor_research_note_en.md)
2. [Claim and evidence ledger](docs/claim_evidence_ledger.md)
3. [Results index](docs/results_index.md)
4. [Reproducibility guide](docs/reproducibility.md)
5. [Code takeover guide](docs/code_takeover_guide.md)
6. [Code map](docs/code_map.md) and [personal scorecard](docs/code_takeover_scorecard.md)
7. [Public release scope](docs/public_release_scope.md)

The sanitized bounded-check artifacts and their SHA-256 manifest are built by
`scripts/build_public_release_artifacts.py` under `results/public_release_v1/`.

## Quick verification

```powershell
git clone https://github.com/rzCayt/error-guided-sft-data-selection.git
cd error-guided-sft-data-selection
python -m pip install -e ".[dev]"
pytest -q

# CPU-only deterministic audits
python scripts/audit_selector_identifiability.py `
  --output-dir results/reproduction/selector_identifiability
python scripts/audit_residual_selector_identifiability.py `
  --output-dir results/reproduction/residual_selector_identifiability
```

The model-aware F1/F2 checks require a CUDA environment and the frozen
Qwen3-1.7B revision. See [the reproducibility guide](docs/reproducibility.md)
before running them.

## Research boundary

Development diagnostics are used for selector design and audit. They are not a
held-out final test set. Tulu-pool H1a is complete, but the 48-candidate
GSM8K-domain boundary check and all downstream LoRA/SFT comparisons remain
unfinished.

## 中文摘要

本项目研究：目标模型在诊断集上的错误，能否形成超越 all-query 分数和训练词元
长度的候选级训练数据效用信号。旧的 metadata selector、residual selector 和八候选
试验仍是负结果；新的 96 候选 Tulu 正式 H1a 则通过了预注册的三项门槛：
partial Spearman 为 0.22775，单侧 1,000 次标签置换 `p=0.07293`，高分组平均效用
高于低分组。该结果只说明可以继续做固定预算的 `B=500` 训练比较，尚不能声称
error-conditioned RDS+ 能提高最终准确率或优于 random/all-query RDS+。
