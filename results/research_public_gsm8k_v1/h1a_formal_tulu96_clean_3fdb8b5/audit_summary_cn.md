# Tulu 96 候选正式 H1a 审计摘要

## 结论

正式 Tulu 候选池 H1a 通过预注册的三项门槛：

| 检查 | 结果 | 门槛 |
|---|---:|---:|
| 控制 all-query 分数和训练词元长度后的 partial Spearman | 0.227749 | ≥ 0.15 |
| 1,000 次固定错误数量标签置换的单侧 p 值 | 0.072927 | ≤ 0.10 |
| error-query 高分 24 条减低分 24 条的平均效用 | 0.002494 | > 0 |

该结果只支持进入预注册的固定预算训练比较，不等于已经证明下游 SFT 准确率提升。

## 输入完整性

- 评分候选：96 条，ID 唯一。
- 效用测量：96 条，ID 唯一，与评分候选一一对应。
- diagnostic query：448 条。
- 真实错误 query：99 条，与 `numeric_correct=false` 的 ID 集合完全一致。
- 96 条候选都有至少一个可监督回答词元。
- 评分、效用和分析均使用固定的数据、模型 revision 与随机种子。

## 独立复核

- 使用 NumPy/SciPy 独立重算 partial Spearman，得到
  `0.22774902568147176`，与正式 artifact 一致。
- 1,000 个置换集合都从 448 条 query 中不放回抽取 99 条，集合内部无重复。
- 共有 72 个置换统计量不低于真实统计量；`+1` 修正后
  `p=(72+1)/(1000+1)=0.07292707292707293`。
- 从保存的 embedding 重算 all-query 与 error-query 分数，最大误差均为 0。
- top 与 bottom 各 24 条、无交集，分组和效用差与 artifact 一致。
- 96 条 token audit 与评分文件中的总词元数、监督词元数全部一致。

## 可复现性

第一次正式效用运行的已跟踪代码没有差异，但仓库根目录存在未跟踪日志，导致
manifest 记录为 dirty。为消除该记录警告，全部 96 条效用在全新 clone 中重新运行，
日志和输出均放在仓库外。

两次效用文件逐字节一致：

`utility_measurements.jsonl`

SHA-256：

`db8b384ede5ca21de95a0108ddda19430c65758cc90b245faf0b0372ab788c75`

最终 utility 和 analysis manifest 均记录：

- git commit：`3fdb8b556bb97bf990e30e8ecd11a7fd809f6536`
- `git_is_dirty=false`
- `git_diff_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

最后一次全仓测试为 `161 passed`，相关文件的 Ruff 检查通过。

## 结论边界

本审计没有证明：

- H1a 信号能推广到 GSM8K 域内候选；
- error-query RDS+ 能在完整 LoRA/SFT 后提高 GSM8K exact match；
- error-query RDS+ 能在训练结果上超过 all-query RDS+ 或 random；
- 结果能直接推广到 Qwen3-4B 或其他模型规模。
