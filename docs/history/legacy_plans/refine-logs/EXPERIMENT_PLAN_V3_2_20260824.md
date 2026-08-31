> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# LLM后训练研究执行计划 v3.2

**日期**：2026-08-24
**当前提交基线**：`064db87`之后的CPU准备分支
**研究边界**：不准备CV和教授邮件；GPU已关机；正式Phase 1冻结。

## 1. 当前不是项目结束

当前只完成了“进入正式训练前的数据、名单和合同层”。已经完成：

- 448×8542完整相似度；
- 8542条近重复簇控制；
- 四方法×四个selection replicate的16份名单；
- targeted-policy gate通过；
- error-conditioning increment gate失败，因此删除`rds_all`完整训练矩阵；
- Phase 1十六格矩阵SHA冻结；
- 16/16 CPU contract审计PASS；
- 私有盲态映射和公开匿名registry冻结。

这些工作证明“实验可以公平地开始”，尚未证明RDS有效。

## 2. 附件与当前协议的优先级

附件v2.1和详细执行版只作为设计参考。用户已批准的v3.1覆盖以下旧设计：

- 第一批使用16格，不恢复旧版12格；
- GSM8K exact-match保持唯一主指标，SVAMP/ASDiv/MultiArith为关键次要指标；
- 监督目标约32000词元，不改成32768；
- 错误条件化信息闸门不事后放宽；
- 当前闸门失败后不运行`rds_all`完整矩阵；
- 第一阶段不加入新selector。

## 3. 必须完成的研究阶段

### R0：CPU运行准备（当前阶段）

必须完成：

1. 冻结16格matrix config和每格selection SHA；
2. 16格contract-only审计；
3. 盲态方法映射、匿名registry和解盲门槛；
4. qualification合同；
5. OOD数据revision、prompt、parser和污染审计冻结；
6. 结果聚合器只能读取`AUDITED_PASS`；
7. 聚合前只输出匿名方法和方差，不输出真实方法名；
8. fresh-clone恢复说明和云端上传清单。

完成标准：CPU测试、Ruff、secret/absolute-path扫描通过，正式训练仍为0格。

### R1：GPU qualification

先用一张同型号4090完成：

- 16样本过拟合；
- LoRA参数/梯度审计；
- adapter保存与重载；
- optimizer、scheduler和RNG恢复；
- 人为中断后的checkpoint resume；
- 128题canary generation；
- parser重新计算一致；
- wall-clock、tokens/s、峰值显存和磁盘记录。

第二张GPU只有用户重新授权后才租用。它只做环境和canary等价性，不开始正式矩阵。两卡qualification未通过前，Phase 1保持冻结。

### R2：Phase 1初步16格

矩阵：

```text
4个selection replicate × 4种方法 × train seed 17 = 16格
```

作用：

- 初步估计总策略效应与共同组成内排序效应；
- 估计selection replicate方差；
- 检查common-mix是否改变旧9/9负结果；
- 为确认矩阵和导师沟通提供初步证据。

限制：不能声称有效、无效或等效。至少12个有效格才达到“可以考虑联系教授”的实验门槛；用户当前仍要求不准备联系材料。

### R3：Phase 2确认矩阵

按v3.1执行60格：

- common-mix主比较：32格；
- free-mix总策略比较：16格；
- BM25、k-center、longest诊断：12格。

必须加入：

- GSM8K 1319；
- SVAMP、ASDiv、MultiArith；
- 一个冻结扰动稳健性数据集；
- 分层bootstrap、混合效应模型、leave-one-replicate-out；
- selection、training和item方差分解。

### R4：候选到集合机制

#### M0 候选效用可靠性

- 固定96个候选；
- 每候选3个adapter seed；
- ICC、排名稳定性和测量误差；
- ICC<0.75则停止集合外推。

#### M1 微集合非加性

- 约120个集合；
- 集合大小1/4/8/16；
- 匹配监督词元和候选出现频率；
- 测量冗余、梯度冲突、来源集中和查询覆盖。

#### M2 确认集合

- 48集合×3训练seed；
- 独立utility set；
- candidate-grouped cross-validation；
- 交互模型相对加性模型样本外Spearman至少提高0.10或R²稳定提高。

### R5：预算与模型泛化

- 8K、32K、128K监督词元；
- 先筛查每预算8格，再按门槛扩展；
- 1.5B机制冻结后，3B–4B最小16格；
- 当前不规划8B正式矩阵。

### R6：投稿准备

至少需要：

- 60格确认矩阵；
- 三个OOD和一个扰动任务；
- Longest、BM25和一个强外部selector；
- M0/M1/M2；
- 第二预算；
- 3B–4B复现；
- 公开代码、原始输出、成本、失败记录和独立claim audit。

## 4. 下一批实际执行顺序

1. 完成剩余CPU运行工具和测试；
2. 冻结OOD数据和统一评估接口；
3. 生成云端qualification包；
4. 用户授权后开一张4090，只跑single-GPU qualification；
5. 审计通过后再申请第二张同型号卡做等价性；
6. 两卡通过后才开始16格；
7. 16格全部审计后盲态统计，再统一解盲；
8. 根据方差和方向决定60格的精确扩展，不改预注册门槛。

## 5. 明确禁止

- 不因当前结果新增selector；
- 不把Phase 0名单gate写成模型效果；
- 不把错误条件化失败解释成RDS必然无效；
- 不在16格中途查看真实方法准确率；
- 不自动启动下一格；
- 不在GPU qualification中消耗正式Phase 1格子；
- 不使用公司数据、代码或结果；
- 不提前制作CV和教授邮件。
