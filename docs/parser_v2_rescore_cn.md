# Parser v2 离线重评分结果

本轮没有重新跑模型，只读取 `results/prompt_rescue/*/prompt_rescue_outputs.jsonl`，用更明确的 parser v2 重新评分。

parser v2 的规则是：优先识别 `final answer`、`final numeric answer`、`final value is` 这类明确答案标记；如果没有命中，再回退到 last-number parser。

## 主要结果

| model | prompt variant | v1 acc | v2 acc | delta | incorrect -> correct | correct -> incorrect |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | current_direct | 0.24 | 0.28 | +0.04 | 4 | 0 |
| Qwen2.5-0.5B | final_answer_only | 0.00 | 0.00 | +0.00 | 0 | 0 |
| Qwen2.5-0.5B | step_by_step_final_answer | 0.00 | 0.00 | +0.00 | 0 | 0 |
| Qwen2.5-0.5B-Instruct | current_direct | 0.10 | 0.10 | +0.00 | 0 | 0 |
| Qwen2.5-0.5B-Instruct | final_answer_only | 0.15 | 0.15 | +0.00 | 0 | 0 |
| Qwen2.5-0.5B-Instruct | step_by_step_final_answer | 0.03 | 0.03 | +0.00 | 0 | 0 |

完整 artifacts：

- `results/prompt_rescue_rescore/parser_v2_summary.csv`
- `results/prompt_rescue_rescore/parser_v2_outputs.jsonl`
- `results/prompt_rescue_rescore/parser_v2_run_metadata.json`

## 解释

parser v2 确实修正了一类真实 parser 风险：base model 在 `current_direct` 下有时先给出正确短句，例如 `The final value is 30.6`，随后继续生成下一个 problem，v1 last-number parser 会抓到后续 problem 里的数字。

但这个修正只带来 4 个样例的净增益，`Qwen/Qwen2.5-0.5B` 最好也只有 0.28，仍低于 0.31 的最低继续线，更低于原先设定的 +0.10 到 +0.15 改善门槛。

## 阶段判断

parser v2 不改变 0.5B prompt rescue 的结论：本阶段应判定为完成但未通过。

因此现在不建议进入 micro-SFT。原因是当前证据显示，0.5B 的主要问题不是一个可由 prompt 或 parser 轻易修复的接口问题，而是基础数值计算、公式执行和任务关系理解能力不足。直接进入 LoRA/SFT 容易把训练预算花在修补一个过弱 base model 上，并且很难形成可信的 selection 结论。

下一步应转向更合适的 base diagnostic：

1. 优先跑 `Qwen/Qwen2.5-1.5B`，因为它仍然小、成本可控，且能直接检验 scaling 是否缓解基础算术崩溃；
2. 如果 1.5B 仍然明显失败，再考虑 `Qwen/Qwen2.5-3B`；
3. 数学专用模型可以作为额外 sanity control，但不应替代通用 base model 主线，否则研究问题会从 data selection 变成数学模型选择。

不能声明：

- 不能说 parser v2 证明 0.5B 可救；
- 不能说 0.5B 已适合进入 LoRA/SFT 主实验；
- 不能说 error-guided selection 已经有效。
