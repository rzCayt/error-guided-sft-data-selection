# Qwen2.5-0.5B 可行性说明

## 结论

`Qwen/Qwen2.5-0.5B` 适合作为本项目第一个探索性真实 base diagnostic 模型。

它足够小，本机已能加载；同时它符合研究设定：先使用 base model 诊断数值推理失败，再研究这些诊断错误能否指导 SFT 数据选择。这个可行性结论不等于 downstream selector 或 LoRA comparison 已经有效。

## 证据

官方模型信息：

- Qwen2.5 系列包含 0.5B 到 72B 的 pretrained 和 instruction-tuned models：https://huggingface.co/collections/Qwen/qwen25
- `Qwen/Qwen2.5-0.5B` model card 标注 32,768 token context length，并说明 base language model 不推荐直接用于对话，更适合 SFT/RLHF/continued pretraining 等 post-training：https://huggingface.co/Qwen/Qwen2.5-0.5B
- Qwen2.5 blog 指向 Hugging Face Transformers 作为常规使用路径：https://qwenlm.github.io/blog/qwen2.5/

本地 smoke test：

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

Prompt：

```text
Problem: A metric starts at 100 and increases by 15%. Final value =
```

观测 continuation：

```text
 115. What is the initial value?
What is the step-by
```

这足以说明模型可以加载，并能生成包含可解析数字的 completion。同时它也说明 diagnostic script 应使用 completion-style prompts、deterministic decoding、较短的 `max_new_tokens` 和稳健 parser，而不是把 base model 当 chat model 使用。

上面的显存观测只来自一个短 prompt，不保证完整 diagnostic batching 或 LoRA training 的显存余量。

## Base vs Instruct 选择

主实验模型：

```text
Qwen/Qwen2.5-0.5B
```

使用它是因为研究问题关注 base model 的 post-training。

可选 sanity model：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

只有当 base model 输出过难解析时，才把 instruct model 用作 pipeline sanity check。不能把 instruct-model 结果混入 base-model SFT claim。

## 实操建议

- 先用 greedy decoding：`do_sample=False`。
- `max_new_tokens` 保持较短，例如 16 到 48。
- 使用让答案自然延续的 completion-style prompt。
- 解析前保存 raw outputs。
- 记录 model id、model revision、tokenizer id、dtype、device、decoding config、parser version 和 seed。
- parse failure 是真实 diagnostic error，不能随意丢弃。

## 风险

0.5B base model 可能很弱，completion 格式也可能不稳定。这对 diagnostic error profiling 是可以接受的，但也意味着导师材料必须聚焦 pipeline 和 controlled comparison。在真实 Random vs Targeted LoRA 结果出现前，不能做性能提升声明。
