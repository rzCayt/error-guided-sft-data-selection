# 如何说明我的项目贡献

## 简短版本

我搭建了一个小型、可复现的研究脚手架，用来研究 base model 的诊断错误能否指导 SFT 数据选择。我的贡献重点是实验设计和可审计 pipeline，而不是声称当前方法已经提升真实模型。

## 我具体做了什么

- 设计并实现了一个确定性的数值推理数据生成器，每条样本都有 solver-verified label。
- 设计了独立 split：训练候选池、base diagnostic、ID test、OOD template test、OOD range test。
- 实现了 diagnostic pipeline，记录预测、解析成功率、数值准确率、输出长度和错误类型。
- 实现了 error-guided data selector 和 matched-random baseline。
- 实现了 bias/leakage audit，包括 split duplicate check 和 targeted/matched overlap check。
- 建立了阶段式 adversarial review workflow，用来在产生真实训练声明前检查泄漏、baseline 不公平和过度表述。
- 整理了 data-efficient instruction tuning、data selection 和 LoRA 相关文献定位。
- 将项目入口文档和研究说明改为中文优先，并保留英文版本用于 GitHub/国际读者。

## 我没有声称什么

- 我没有声称 error-guided selection 已经优于 matched random。
- 我没有声称已经完成真实 LoRA 训练结果。
- 我没有声称合成任务能证明广义数学推理能力。
- 我没有声称当前 selector 是最终方案；下一版应该加入 error-type-aware selector 和 ablation。

## 可以这样说明这个研究

```text
我设计并实现了一个用于 error-guided SFT data selection 的可控 pilot pipeline。这个项目重点不是先追求漂亮结果，而是把实验流程做严谨：确定性数据生成、solver-verifiable labels、严格 split discipline、matched-random baseline、leakage audit、bias audit，以及每个阶段进入下一步前的 adversarial review。当前阶段已经验证了研究框架和本地 pipeline，下一步是用 Qwen2.5-0.5B 替换 simulated diagnostic，收集真实 base model 错误，再测试错误诊断驱动的数据选择是否比 metadata-matched random sampling 提供额外信号。
```

## 研究表述重点

应该强调：

- 研究问题清楚。
- comparison design 可控。
- baseline 公平性和泄漏风险被显式处理。
- 项目可复现、可审计。
- 愿意报告负结果或不显著结果。
- 已准备进入真实 base diagnostic 和 LoRA 对比。

避免说：

- “我做了一个金融大模型。”
- “模型已经提升了。”
- “Targeted 方法已经被证明有效。”
- “这个数据集证明了模型推理能力。”

## 一分钟口头版本

```text
我在做一个小型 post-training 研究项目，问题是 base model 的诊断错误能不能指导 SFT 数据选择。具体来说，我先让 base model 在独立诊断集上暴露错误，再把错误画像转成 candidate pool 的选择信号，然后和严格 matched random baseline 在相同预算下比较。我已经实现了 generator、solver、split discipline、parser、error taxonomy、targeted/matched-random selection 和 audit reports。项目还设置了一个只读审核流程，专门检查 leakage、baseline fairness 和 overclaiming。当前阶段还不声明训练增益，下一步是用 Qwen2.5-0.5B 跑真实 base diagnostic。
```

## 如果被问“你的核心贡献是什么”

可以回答：

```text
我的核心贡献是把一个比较宽泛的 data selection 想法变成可复现的实验协议：solver-verifiable task setup、严格 split discipline、diagnostic error profiling、公平 matched-random comparison，以及防止 premature claims 的 audit gate。后续真正的研究贡献要看真实模型 diagnostic 是否能证明 error-type-aware selection 在 matched random 之上带来额外信号。
```
