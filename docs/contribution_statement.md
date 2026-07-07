# 如何说明我的研究贡献

## 简短版本

我不是只做了一个代码 demo，而是在做一个可复核的小型研究：先让 base model 在独立诊断集上真实出错，再分析这些错误能不能成为 SFT 数据选择信号。

当前最重要的贡献不是“模型已经提升”，而是把研究问题拆成了可检查的步骤，并且保留了原始输出、错误画像和不能下结论的边界。

## 我具体完成了什么

- 设计了一个 solver-verifiable 的数值推理数据集，每条样本都有确定答案。
- 固定了 `candidate_pool`、`dev_diagnostic` 和 ID/OOD test split，避免把诊断集和测试集混用。
- 用 `Qwen/Qwen2.5-0.5B` 跑了第一轮真实 base diagnostic，保存 raw outputs、model/tokenizer revision、prompt、decoding config 和 parser version。
- 生成了 `results/real_error_profile.csv`，并用真实错误画像重跑 strong baseline selection artifacts。
- 新增 parser/error audit，把错误初步分成推理错误、parser/格式风险、prompt/题目理解风险。
- 保留 matched random、stratified random 和 metadata-hard baseline，用来防止后续把普通难度重采样误说成方法提升。

## 我的研究判断在哪里

这个阶段最能体现研究能力的部分，是判断错误画像是否可信：

- `weighted_aggregation` 全错，可能是模型不会加权，也可能是题目措辞或输出解析造成的现象。
- `ratio_change` 经常输出多个数字和等式，说明 last-number parser 可能影响结论。
- `multiplicative_relation` 在 easy 上表现好，但 hard 上全错，提示难度分层可能是真信号，但仍需要人工样例复核。

因此我不会直接进入 LoRA 训练，而是先做人工错误复核。这个决策本身就是研究判断：先确认信号质量，再决定是否训练。

## 如何说明 AI 的角色

可以直接说明使用了 AI，但要区分角色：

```text
我把 AI 当成 coding assistant 和 review assistant。研究问题、证据边界、强基线要求、是否暂缓 LoRA、以及哪些错误需要人工复核，是我主导判断的部分。AI 帮我把这些判断落实成脚本、文档和可复现产物。
```

这样的说法比回避 AI 更可信，也更符合现在的研究工作方式。

## 一分钟口头版本

```text
我现在做的是一个小型 post-training 研究，问题是 base model 的诊断错误能不能指导 SFT 数据选择。我已经用 Qwen2.5-0.5B 跑了第一轮真实诊断，100 条 dev diagnostic 里数值准确率是 0.21。现在我没有直接说方法有效，而是在分析错误画像：哪些是真推理错误，哪些是 parser 或输出格式问题，哪些可能是题目表达导致的误解。下一步我会人工复核 20-30 条错误样例，再决定是否进入 selection bias audit 和 LoRA 对比。
```

## 如果被问“你在项目中是什么身份”

可以回答：

```text
我不是数据提供者，也不是只给 AI 提意见的人。我在这个项目里的角色是研究问题负责人和错误分析者：我负责定义问题、控制 split 和 baseline、公平解释结果，并判断下一步实验是否有意义。
```

## 避免的说法

- “模型已经提升了。”
- “Targeted selection 已经被证明有效。”
- “我做了一个金融大模型。”
- “这个结果已经能说明 LoRA 有用。”
