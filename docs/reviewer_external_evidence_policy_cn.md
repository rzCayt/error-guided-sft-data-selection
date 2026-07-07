# 审核线程外部资料核验策略

审核线程不能只读本仓库。每个会影响研究设计、模型选择、训练配置、结果表述或导师材料的阶段，都必须做外部资料核验。

## 必须搜索的情况

以下阶段必须搜索资料：

- `workflow_upgrade`：核对流程、评分、review gate 是否覆盖研究可信度风险。
- `real_base_diagnostic`：核对模型卡、推理限制、base vs instruct 使用边界。
- `selection_bias_audit`：核对 data selection 文献对 baseline、公平比较、selection signal 的要求。
- `lora_comparison`：核对 LoRA/SFT 训练配置、可复现性和对比公平性。
- `professor_summary`：核对材料是否把现有证据说成超出文献和实验支持的结论。

## 来源优先级

优先使用下面来源，低优先级来源只能作为补充：

1. 论文原文、arXiv、ACL Anthology、官方技术报告。
2. 官方模型卡、官方文档、官方 GitHub。
3. 研究机构技术博客。
4. 第三方博客、社交媒体、新闻，只能用于发现线索，不能作为核心结论依据。

## 固定检索任务

审核线程必须至少做三类检索：

- 方向检索：当前阶段是否仍符合 data selection / SFT / LoRA 方向。
- 风险检索：文献或模型卡是否提示 baseline、公平性、泄漏、可复现性或模型限制风险。
- 反例检索：是否有证据说明当前计划过弱、不可比、不可复现或容易误导。

## 输出要求

审核响应必须包含 `检索记录` 和 `外部资料核验`：

- `检索记录` 至少 3 条，记录 query、用途、是否影响 verdict。
- `外部资料核验` 至少 3 条，记录 source type、URL、关键主张、对本项目的影响、是否构成 blocker。
- 如果允许进入下一阶段，至少 2 条外部资料必须是一手来源。
- 每条资料都必须说明“如何改变或约束当前计划”，不能只罗列链接。

## 当前推荐 seed sources

- LESS: Selecting Influential Data for Targeted Instruction Tuning: https://arxiv.org/abs/2402.04333
- Take the Essence and Discard the Dross: A Rethinking on Data Selection for Fine-Tuning Large Language Models: https://aclanthology.org/2025.naacl-long.336/
- Qwen/Qwen2.5-0.5B model card: https://huggingface.co/Qwen/Qwen2.5-0.5B
- Qwen2.5 blog: https://qwenlm.github.io/blog/qwen2.5/
- LoRA Without Regret: https://thinkingmachines.ai/blog/lora/

## 不能通过的情况

出现任一情况，审核线程必须给出 blocker 或 conditional fail：

- 没有外部检索，却评价模型选择、训练配置或 research claim。
- 只引用二手博客，却没有论文、模型卡或官方文档。
- 引用了资料但没有说明它如何约束当前阶段。
- 文献要求公平 baseline 或设置一致，而审查包没有对应 artifact。
- 模型卡或官方文档限制与项目使用方式冲突，但主线程没有处理。
