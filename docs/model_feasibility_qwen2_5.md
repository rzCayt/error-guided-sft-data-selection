# Qwen2.5-0.5B Feasibility Note

## Verdict

`Qwen/Qwen2.5-0.5B` is feasible for this project as the first exploratory real base diagnostic model.

It is small enough to load on the current local machine and it matches the research framing: use a base model, diagnose its numerical-reasoning failures, then study whether diagnostic errors can guide SFT data selection. This is not evidence that the downstream selector or LoRA comparison works.

## Evidence

Official model information:

- The Qwen2.5 collection includes pretrained and instruction-tuned models from 0.5B to 72B parameters: https://huggingface.co/collections/Qwen/qwen25
- The `Qwen/Qwen2.5-0.5B` model card lists a 32,768 token context length and says base language models are not recommended for conversations; instead, they are intended for post-training such as SFT/RLHF/continued pretraining: https://huggingface.co/Qwen/Qwen2.5-0.5B
- The Qwen2.5 blog points to Hugging Face Transformers as the simple usage path: https://qwenlm.github.io/blog/qwen2.5/

Local smoke test:

```text
Command: python scripts/run_qwen_smoke.py
Artifact: results/qwen2_5_0_5b_smoke.json
Model: Qwen/Qwen2.5-0.5B
Revision: 060db6499f32faf8b98477b0a26969ef7d8b9987
Device: NVIDIA GeForce RTX 5060 Laptop GPU
CUDA available: true
Transformers: 4.57.2
PyTorch: 2.8.0+cu128
Observed CUDA memory after one short generation: about 958 MB allocated, 960 MB peak allocated
```

Prompt:

```text
Problem: A metric starts at 100 and increases by 15%. Final value =
```

Observed continuation:

```text
 115. What is the initial value?
What is the step-by
```

This is enough to show the model loads and can produce a parseable numeric completion. It also shows why the diagnostic script should use completion-style prompts, deterministic decoding, short `max_new_tokens`, and robust parsing rather than chat-style expectations. The memory observation is only for one short inference prompt, not a guarantee about full diagnostic batching or LoRA training headroom.

## Base vs Instruct Choice

Primary model:

```text
Qwen/Qwen2.5-0.5B
```

Use this for the main research question because it is a base model and keeps the post-training setup clean.

Optional sanity model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Use only as a pipeline sanity check if the base model outputs are too hard to parse. Do not mix instruct-model results with base-model SFT claims.

## Practical Guidance

- Use greedy decoding first: `do_sample=False`.
- Keep `max_new_tokens` short, such as 16 to 48.
- Use prompts that make the answer a direct continuation.
- Store raw outputs before parsing.
- Record model id, model revision if available, tokenizer id, dtype, device, decoding config, parser version, and seed.
- Treat failures to parse as real diagnostic errors, not as rows to discard.

## Risk

The 0.5B base model may be weak and sometimes format completions poorly. That is acceptable for diagnostic error profiling, but it means professor-facing claims must focus on the pipeline and controlled comparison until real Random vs Targeted LoRA results exist.
