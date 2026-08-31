> **Historical snapshot.**
>
> This document records an earlier research stage and is not the current result.
>
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# 预算等价 v3：当前实施状态与下一步

## 已经完成

- 从正式提交`6d5bcea`建立独立干净开发副本；
- 保留旧9/9负结果和旧运行器；
- 实现四种核心方法的统一预算求解；
- 实现真实查询bootstrap，而不是伪造不同seed；
- 实现500条、32K回答监督词元和common-mix配额；
- 复用并验证response-token weighted loss；
- 把1000次样本曝光严格分成64个优化步骤；
- 实现相似度导出、近重复聚类、信息性闸门、16格矩阵和逐格审计；
- 完整CPU测试265项通过，新增相关测试23项通过，ruff通过。

## Phase 0当前结论

- 单张RTX 4090D已完成7个查询块和67个候选块，深度哈希审计通过；
- 完整相似度为`448×8542`，SHA-256为`994583a61022f84c5128e42d71956fc44f7b9bd987b34b0ec7f0bb249ca58a11`；
- 近重复清单覆盖8542条候选、7869个簇，SHA-256为`a402b8ad118f5b2d9b90b1b2ea1679c302a43955208454b0617fff5a50c68fd1`；
- 四种方法×四个selection replicate的16份名单已生成，独立审计PASS；
- targeted policy gate通过：名单不同、分数非恒定、强制入选比例0、common-mix最小自由度9；
- error-conditioning increment gate失败：最小Top-500更换率只有7%，完整排序Spearman最高0.998，虽稳定性中位数Jaccard为0.871；
- 因此不运行`rds_all`完整训练矩阵，也不声称二元错误标签产生新增选择政策；
- 这仍允许按预注册方案比较`rds_error`与`random`的16格核心矩阵。

printf 'progress='; wc -l "$RUN"/evaluation/workers/test_shard*/raw_outputs.jsonl 2>/dev/null | tail -1 | awk '{print $1+0}'LoRA下游有效性结论。

## 下一次执行顺序

1. 把Phase 0修复提交和冻结配置同步到公开分支；
2. 下载并核验Phase 0完整归档；
3. 保持正式Phase 1训练冻结；
4. 用户确认第二张同型号4090后，进行两卡运行环境等价性检查；
5. 完成16样本过拟合、adapter保存/重载/恢复和128题canary；
6. qualification全部通过后，才冻结并启动Phase 1的16格。

CV和教授邮件，不租第二张GPU，不启动正式LoRA矩阵。
