# Qwen2.5-0.5B prompt/parser rescue 结果

本轮只在 `dev_diagnostic` 上做诊断，不使用 test split，不产生 LoRA、SFT 或 error-guided selection 有效性结论。

## 运行设置

- Base model: `Qwen/Qwen2.5-0.5B`
- Control model: `Qwen/Qwen2.5-0.5B-Instruct`
- Split: `dev_diagnostic`
- 每个模型：100 条样例 × 3 个 prompt variants
- Decoding: greedy, `max_new_tokens=64`
- Parser: `parse_numeric_final_answer_v1_fallback_last_number`

输出目录：

- `results/prompt_rescue/Qwen_Qwen2.5-0.5B/`
- `results/prompt_rescue/Qwen_Qwen2.5-0.5B-Instruct/`

## 主要结果

| model | prompt variant | accuracy | parse success | gain vs 0.21 |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | current_direct | 0.24 | 1.00 | +0.03 |
| Qwen2.5-0.5B | final_answer_only | 0.00 | 0.00 | -0.21 |
| Qwen2.5-0.5B | step_by_step_final_answer | 0.00 | 0.00 | -0.21 |
| Qwen2.5-0.5B-Instruct | current_direct | 0.10 | 1.00 | -0.11 |
| Qwen2.5-0.5B-Instruct | final_answer_only | 0.15 | 1.00 | -0.06 |
| Qwen2.5-0.5B-Instruct | step_by_step_final_answer | 0.03 | 1.00 | -0.18 |

## 解释

`Qwen/Qwen2.5-0.5B` 的最好结果只有 0.24，相比原始 0.21 只提升 0.03，没有达到预设的 +0.10 到 +0.15 继续条件。

`Qwen/Qwen2.5-0.5B-Instruct` 作为控制模型没有超过 base model。这个结果说明当前低正确率不只是 instruction following 或输出格式问题，更可能来自 0.5B 模型本身的基础数值计算和公式执行能力不足。

base model 在 `final_answer_only` 和 `step_by_step_final_answer` 下 parse success 为 0，说明这两个 prompt 不适合作为 base completion model 的有效提升证据。它们更多暴露了 prompt 与 base 模型接口不匹配。

## Parser 观察

抽样检查发现，base model 有时会先写出正确短语，例如 `The final value is 30.6`，随后继续生成另一个 problem，导致 last-number parser 抓到后续数字。只读重评分显示，如果把 `final value is` 作为明确答案标记，`current_direct` 可能从 0.24 提升到约 0.28。

这个 parser 修正值得记录，但仍不足以改变阶段结论：0.5B prompt/parser rescue 没有通过。

## 阶段结论

本轮不支持继续把 0.5B prompt engineering 作为主线。更合理的下一步是：

1. 做一个很小的 parser v2 修正，识别 `final value is` 这类明确答案标记；
2. 用同一 raw outputs 重新生成 summary，确认 parser 修正上限；
3. 如果仍低于 0.31 到 0.36 的继续条件，停止 0.5B prompt rescue；
4. 转向 1.5B/3B 或数学专用模型，验证错误画像是否仍有 data selection 价值。

不能声明：

- 不能说 prompt rescue 提升了模型推理能力；
- 不能说 0.5B 已经适合进入 LoRA/SFT 主实验；
- 不能说 error-guided selection 已经有效。
