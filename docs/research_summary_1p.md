# 一页研究摘要

## 项目

**基于错误诊断的数据选择：面向小型数值推理语言模型的高效 LoRA SFT**

英文题目：

```text
Error-Guided Data Selection for Data-Efficient LoRA SFT in Small Numerical Reasoning Language Models
```

## 研究问题

在相同样本预算和训练预算下，base model 的诊断错误能否指导 SFT 数据选择，并比 matched random selection 更有效？

## 方法概述

项目构建了一个可控的合成数值推理 benchmark，每条样本都有 deterministic solver 生成的可验证答案。流程先让小型 base language model 在独立的 `dev_diagnostic` split 上作答，再把失败解析为 error taxonomy，例如算术错误、公式错误、单位/尺度错误、时间顺序错误、解析失败和变量绑定错误。

随后，项目根据错误画像从独立 `candidate_pool` 中选择 SFT 样本。对照组是 matched-random baseline，会匹配任务族、难度、答案量级和推理长度。如果严格 stratum matching 需要与 targeted subset 重叠，系统会报告 overlap，并把它作为 baseline 独立性限制，而不是隐藏。

锁定的 ID/OOD test split 不参与策略调整。

## 当前证据

仓库已经包含可复现 generator、solver、模拟 diagnostic path、selection policies、bias audit 和 workflow validator。模拟路径只是为了在没有模型/GPU 的环境中验证 pipeline，不支持任何真实训练增益声明。

## 下一步

运行 `Qwen/Qwen2.5-0.5B` 的真实 base diagnostic，并在 GPU 环境中做 LoRA smoke test。随后在 B128/B256 预算下比较 Base、Matched Random 和 Targeted。

## 研究价值

该项目展示的是 post-training 研究中的基础能力：数据构造、诊断评估、数据选择、baseline 公平性、泄漏控制、实验记录和克制汇报。更重要的是，它把一个容易被讲成概念的方向落到了可检查的研究协议上：哪些证据已经存在、哪些结论还不能说、下一步实验需要满足什么最低标准，都被明确写入 workflow。
