# 文献综述与研究定位

## 为什么 data selection 是合理的 RA 项目

近年的 fine-tuning data selection 研究通常把数据选择看成预算约束下的优化问题：目标不是简单地少用数据，而是在固定样本或 token 预算下选择更能提升目标能力的数据。

本项目沿用这个 framing，但把规模缩小到更可审计的环境：数值推理任务是合成的、答案可由 solver 验证、目标能力明确、选择子集会与 matched-random baseline 比较。

## 相关工作

**LESS: Selecting Influential Data for Targeted Instruction Tuning.** LESS 把 targeted instruction tuning 定义为选择能诱导目标能力的数据，例如 reasoning。它使用 low-rank gradient similarity 选择 influential examples，并报告小规模选中子集可以在下游任务上超过更大的训练集。本项目不实现 gradient influence，而是借用 targeted-capability framing，用 diagnostic error profile 作为更便宜、更透明的选择信号。来源：https://arxiv.org/abs/2402.04333

**RL-Guided Data Selection for Language Model Finetuning.** 该工作把 fine-tuning data selection 建模为 budget-constrained optimization，并用 proxy reward 学习选择策略。本项目与它的联系主要是概念层面：error-guided selection 是更简单、可解释的策略，适合作为 RA 申请前的第一版项目。来源：https://arxiv.org/abs/2509.25850

**Take the Essence and Discard the Dross.** 这篇 NAACL 2025 review 指出，data-selection 研究很难比较，原因之一是实验设置差异很大，因此需要结构化比较 efficiency 和 feasibility。这个提醒直接约束本项目：在解释任何提升前，candidate pool、budget、test set 和 baseline 必须固定。来源：https://aclanthology.org/2025.naacl-long.336/

**LoRA Without Regret.** 这篇文章支持把 LoRA 作为较低成本的 post-training 工具，但前提是训练设置被仔细控制。本项目使用 LoRA 是为了降低实验成本，不是为了声称 low-rank updates 总能替代 full fine-tuning。来源：https://thinkingmachines.ai/blog/lora/

**LIMA: Less Is More for Alignment.** LIMA 说明，当 base model 已包含广泛知识时，小而高质量的 SFT 集合可能教会 response behavior。这支持“数据质量和选择很重要”的动机，但 LIMA 不是 data-selection algorithm，也不能验证本项目的 error-guided policy。来源：https://arxiv.org/abs/2305.11206

**AlpaGasus: Training A Better Alpaca with Fewer Data.** AlpaGasus 用质量过滤减少 instruction data，并报告更快/更强的 instruction tuning。它适合用于 data-centric instruction tuning 的定位；本项目不同之处是使用 solver-verifiable numerical tasks 和 base-model diagnostic failures，而不是强外部 LLM judge。来源：https://arxiv.org/abs/2307.08701

## 本项目想补的空白

很多强 data-selection 方法需要梯度、学习式选择策略或外部 judge。本项目探索更便宜的 diagnostic route：用 base model failure profile 选择 solver-verifiable SFT examples。

这个贡献只有在两个条件下才可信：

- 与 matched random 的比较足够严格。
- diagnostic signal 被证明提供了 task-family/difficulty resampling 之外的信息。

## 文献带来的风险提示

- 如果 selected examples 只是复制 task-family imbalance，这不是有意义的 error-guided method。
- 如果 random baseline 更弱或匹配不足，targeted gain 可能只是 baseline artifact。
- 如果合成模板太窄，test performance 可能测到模板记忆，而不是 reasoning。
- 如果 simulated diagnostics 被写成真实模型证据，项目可信度会下降。

## 近期研究定位

第一个有质量的 milestone 不是大结果，而是一个干净的小实验：在 B128/B256 预算下，检验 diagnostic errors 是否提供有效选择信号，并配套严格 split discipline 和能经受怀疑者审查的 baseline。
