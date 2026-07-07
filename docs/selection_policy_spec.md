# 数据选择策略规范

## Error-Guided Policy

输入：

- `data/samples/candidate_pool.jsonl`
- `results/error_profile_v0.csv`

选择器根据诊断弱点画像给候选样本打分：

1. 诊断准确率更低的任务族获得更高权重。
2. 失败更多的难度 bucket 获得更高权重。
3. 失败更多的答案量级和推理长度 bucket 获得更高权重。
4. 对每个 stratum 设置上限，通常不让 targeted subset 占据某个匹配 stratum 的一半以上，从而尽量给 matched random 留出同 stratum 样本。
5. 使用基于 `id` 和 seed 的稳定 hash 做 deterministic tie-break。

选择器不能使用 test labels、test predictions 或 test metrics。

## Matched Random Policy

Matched random 必须尽可能保持与 targeted subset 相同的分布：

- 任务族
- 难度 bucket
- 答案量级 bucket
- 推理长度 bucket

在每个 stratum 内，matched random 使用确定性随机 seed 采样。Targeted selector 会通过 stratum cap 尽量避免把同一 stratum 的样本全部取走，因此 matched random 通常可以避免与 targeted 重叠。

如果某个 stratum 中非 targeted 样本不足，matched random 允许与 targeted subset 重叠，而不是破坏 stratum matching。这样 baseline 的分布更公平，但 overlap count 必须报告，因为重叠会降低两个训练集的独立性。

## Bias Audit

Audit 需要报告 targeted 与 matched-random selection 的分布差异：

- matching stratum count
- task family count
- difficulty count
- mean answer magnitude
- mean reasoning length
- targeted/matched overlap rate

如果差异很大，后续比较可能只是 distribution shift，而不是更好的数据选择信号。此时不能直接声明 targeted selection 更有效。
