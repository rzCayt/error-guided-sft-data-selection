> **Historical snapshot.**
>
> This document records an earlier research stage and is not the current result.
>
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# 给 GPT Pro 的 v8 对抗复审提示词

请把压缩包视为“尚未启动正式 GPU 矩阵的研究预备包”，不要默认接受 Codex 的设计。请以严格论文审稿人、实验复现工程师和 RA 导师三种角色互相反驳，重点寻找能改变结论的错误。

请先独立核验源码、配置、manifest 和测试，不以 README 的自述代替证据。随后回答：

1. 24 个全新 common-mix cell 是否真正消除了历史 seed17 与新硬件/代码时期共线？
2. 每个 method×list 的三个种子是否只改变 seed 驱动的顺序、RNG 和优化路径；样本、token、label 和训练配置是否严格不变？
3. canonical runtime manifest 是否是唯一权威；是否仍存在脚本可绕过或引用旧配置？
4. 双 4090D 推理桥接和完整训练锚点能否充分判断两台新机器可合并？数值阈值是否过松或过严？
5. crossed bootstrap 是否保留了全局 training-seed block、方法内名单抽样和共享 item 抽样？
6. +1pp、方向性、等效和退化规则是否表述正确？4 lists×3 seeds 实际能支持哪些精度？
7. 每 worker 独立状态目录、attempt、resume、原子成包和单 GPU 进程是否仍可能污染或重复结果？
8. 24 格运行是否是当前 RA 价值/算力成本最优选择？若不是，请给出更小但更有信息量的替代设计，并说明失去什么结论。
9. 这项工作作为 RA 试做、技术报告、workshop/TMLR 或顶会预备稿分别还缺什么？
10. 请给出明确裁决：`GPU qualification 可开始 / 不可开始`，以及 `training anchor 通过后 24 格可自动开始 / 仍需人工复审`。

请特别反驳以下潜在自我辩护：

- “测试都过了，所以实验科学上一定有效”；
- “三个 seed 足够证明等效”；
- “RDS 四份高重合名单等于四个独立重复”；
- “两张同型号卡天然数值等价”；
- “旧 seed17 和新 seed17 语义接近就能放进同一主分析”；
- “完整跑完 24 格就足以发顶会”。

输出格式：P0 阻塞、P1 重要问题、可接受设计、最强反驳、RA 价值、论文价值、分支决策和最终 go/no-go。不要为了让项目显得好看而降低证据门槛。
