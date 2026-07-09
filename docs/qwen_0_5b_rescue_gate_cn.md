# Qwen2.5-0.5B 推理救援门槛

这个阶段只回答一个问题：`Qwen/Qwen2.5-0.5B` 的低正确率，是否能被更清楚的 prompt 和更稳健的答案解析明显改善。

它不是 LoRA 实验，也不是 error-guided selection 已经有效的证据。

## 为什么先做这个

第一轮真实 diagnostic 的数值正确率是 0.21。人工复核表显示，很多错误非常基础：百分比变化算错、乘法算错、加权聚合公式用错、时间顺序约束没有正确执行。继续扩大人工标注的收益不高，下一步应该先判断这些错误是不是 prompt/parser 问题。

## 运行方式

基础模型：

```powershell
python scripts/run_prompt_rescue_diagnostic.py --model Qwen/Qwen2.5-0.5B --max-new-tokens 64
```

instruction-tuned 控制模型：

```powershell
python scripts/run_prompt_rescue_diagnostic.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 64
```

快速 smoke test：

```powershell
python scripts/run_prompt_rescue_diagnostic.py --model Qwen/Qwen2.5-0.5B --limit 4 --prompt-variants current_direct,final_answer_only
```

输出位置：

- `results/prompt_rescue/<model>/prompt_rescue_outputs.jsonl`
- `results/prompt_rescue/<model>/prompt_rescue_summary.csv`
- `results/prompt_rescue/<model>/prompt_rescue_error_profile.csv`
- `results/prompt_rescue/<model>/prompt_rescue_run_metadata.json`

## 通过和止损

通过条件：

- dev accuracy 相比已记录的 0.21 提升至少 0.10 到 0.15；或
- 明确证明主要问题是输出格式/parser，而不是模型不会算。

止损条件：

- prompt/parser 改动后提升小于 0.05；
- 错误仍主要是基础计算和公式错误；
- `Qwen2.5-0.5B-Instruct` 明显更好，但 base 模型无法通过 prompt 恢复。

## 后续决策

如果 0.5B 有清楚改善，再进入小规模 LoRA/SFT，并且必须比较 no-train base、random、stratified random、targeted/error-guided。  
如果 0.5B 仍然低级错误很多，就把它作为 negative diagnostic，转向 1.5B/3B 或数学专用模型验证。
