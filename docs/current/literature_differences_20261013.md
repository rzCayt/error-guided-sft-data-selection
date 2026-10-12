# 文献方法差异表

整理日期 2026-10-13；复用上一轮截至 2026-10-12 已核对的原始来源。这是候选差异表，不是系统文献综述或“无人做过”的证明。没有调用新的 Claude 审稿；此前异常网页返回不计入阅读证据。

| 原始工作 | 已核对的方法或对象 | 对本项目的约束 | 候选差异与仍需检查 |
|---|---|---|---|
| [Unified Data Selection for LLM Reasoning](https://arxiv.org/html/2605.22389v1) | 方法段、RL 选择与成本附录：高熵 token 的汇总分数用于轨迹选择；RL 对正轨迹排序选择，对负轨迹随机选择 | 已覆盖 SFT/RFT/RL 轨迹筛选、长度/难度基线与评分成本讨论；不能把这些包装成空白 | 当前“提示+示范”的一步更新价值测量与生成轨迹筛选不同，但固定起点测量误差是否被研究仍需全文与引用链核对 |
| [Student-in-the-Loop CoT Distillation via Generation-Time Selection](https://arxiv.org/html/2604.02819v1) | 生成时选择教师分段续写，使用学生困惑度；冷启动格式对齐及教师规模/家族对照 | 使用更大教师和学生适配性选择本身不新；不能默认教师更大就更适合学生 | 本项目优先研究估值误差与集合收益，若引入教师需证明新增数据成本和匹配基线公平 |
| [Which Reasoning Trajectories Teach Students to Reason Better?](https://aclanthology.org/2026.acl-long.1950/) | 摘要与元数据核对：token 排名与负对数似然组成 informative-alignment 指标，跨师生配对 | 已有学生适配性与信息量指标；评分方向和完整实现尚需正文复核 | 尚不能认定其未研究重复测量噪声或集合效应，也不能直接把该指标当复现完成的基线 |
| [Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation](https://aclanthology.org/2026.acl-short.49/) | 摘要核对：针对学生弱技能选数据并使用分解后的推理过程 | 错误/弱技能驱动数据选择已有近邻工作 | 当前错误条件化相对 all-query 的增量信息未获证明，需明确对照；全文实现仍待复核 |
| [Fast and Effective On-policy Distillation from Reasoning Prefixes](https://aclanthology.org/2026.findings-acl.1276/) | 摘要核对：学生生成前缀，教师给 token 监督，提前终止与前缀损失 | 学生状态参与蒸馏与节省生成成本已有方法 | 训练后局部排序转移不等于 on-policy 蒸馏；需进一步核对动态采样及总成本对照 |
| [EULoInf](https://aclanthology.org/2026.findings-acl.1839/) | 摘要、影响近似相关方法与消融片段：预测熵和梯度方向/验证前瞻用于影响估计 | 熵、梯度/影响信息及轻量排序并非空白；需要简单的梯度方向与大小基线 | 回答预测的不确定性与“反复测同一效用的误差”不同，但概念区分本身不证明创新 |

优先比较单位（题目、带答案样本、轨迹或集合）、估值状态、评分目标、重复测量、下游验证和全成本。隐藏状态单 rollout 的 RLVR 选择也是待继续精读的近邻：[原始来源](https://arxiv.org/html/2605.28631v1)。未精读的方法不填造细节。

三个候选目前都只是待证实假设。先做可靠性，是为了让后续结论可解释和减少浪费；不是已经证明它有顶会创新。缺失的正式查新工具和独立原始实验审计仍是后续证据工作。
