# Error-Guided SFT Data Selection

This project asks a narrow question: can diagnostic failures from a target
language model identify training examples that are unusually useful for
post-training?

## Status: frozen negative result

The current evidence does **not** justify an SFT effectiveness experiment.

- The first selector is not identifiable beyond the metadata already used by
  an exact matched-random baseline.
- A residual operation-aware selector remains constant after static operation
  metadata is fixed.
- An eight-candidate model-aware feasibility pilot fails one preregistered
  effect-size gate (`0.0193 < permutation p90 0.0241`).
- No completed result shows that targeted selection beats random selection,
  improves LoRA/SFT, or predicts candidate-level training utility.

This is a useful stopping result: before spending compute on an 18-run SFT
matrix, the project must first show that its score contains candidate-specific
signal beyond simple matching variables.

## Verified evidence

| Check | Scope | Result | Interpretation |
|---|---:|---:|---|
| Original selector identifiability | 500 candidates, budget 128 | Fail | Score is fully controlled by matched metadata |
| Residual selector identifiability | 500 candidates, budget 128 | Fail | Score is still static at the operation-signature level |
| Model-aware feasibility | 8 candidates | Fail | Representation is computable, but the effect gate is not cleared |
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
held-out final test set. H1a (candidate utility) and all LoRA/SFT comparisons
remain future work.

## 中文摘要

本项目研究：目标模型在诊断集上的错误，能否形成超越简单匹配变量的候选级
训练数据效用信号。当前得到的是可复核的负结果，而不是 SFT 提升结果：已有
metadata selector、residual selector 和八候选 model-aware 小实验均未达到继续
训练的门槛。因此当前正确动作是暂停大规模 LoRA/SFT，先验证候选级分数是否
真的能预测训练样本效用。
