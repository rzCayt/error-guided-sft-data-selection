# Third-party data, models, and fixtures

The MIT license in this repository applies to original project code and documentation only. It does not replace the licenses of upstream models, datasets, tokenizers, or benchmark materials.

| Asset | Frozen identity or source | License status | Public-repository policy |
|---|---|---|---|
| Qwen2.5-1.5B Base and tokenizer | `Qwen/Qwen2.5-1.5B`, revision recorded in `configs/frozen/public_gsm8k_v1.json` | Apache-2.0 | No model weights are redistributed. The small tokenizer fixture retains upstream metadata. |
| GSM8K | `openai/gsm8k`, frozen revision in `configs/frozen/public_gsm8k_v1.json` | MIT | Raw dataset text is not redistributed in the public result summary. |
| Tulu v2 processed candidate pool | `Harvard-DCML/tulu-v2-197K-processed`, frozen revision in `configs/frozen/public_gsm8k_v1.json` | Dataset card did not specify a license when the protocol was frozen | Raw candidate text is not redistributed. Public artifacts use permitted IDs, hashes, aggregate statistics, and configurations. |
| SVAMP | Frozen source recorded in `configs/frozen/budget_equivalent_ood_v1.json` | MIT | Raw benchmark text is not included in the public summary. |
| ASDiv | Frozen source recorded in `configs/frozen/budget_equivalent_ood_v1.json` | CC BY-NC 4.0 | Raw benchmark text is not included in the public summary. Use remains subject to the upstream non-commercial license. |
| MultiArith | Frozen source recorded in `configs/frozen/budget_equivalent_ood_v1.json` | Upstream license was not specified in the frozen configuration; manual verification is required before redistribution | Raw benchmark text is not redistributed. |

The exact dataset and model revisions are part of the frozen configuration. Users are responsible for reviewing upstream terms before downloading or using any third-party asset.

