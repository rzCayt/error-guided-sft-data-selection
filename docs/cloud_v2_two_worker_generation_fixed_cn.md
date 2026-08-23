# Cloud-v2 同卡双 worker 生成校准（固定入口）

## 研究边界

该方案只测量一张 GPU 上两个 `batch=1` 生成进程的工程吞吐，不是并发正式训练，
不进入 selector 效果结论。两个进程各自加载同一个 base model 和同一个 adapter，可能因
合计显存过高而 OOM；OOM 或任一 worker 非零退出时禁止创建 PASS 合并结果。

## 为什么使用 fixed 入口

运行期目录固定在 Git 已忽略的：

```text
.aris/compute/cloud_v2_two_worker_generation_runs_v1/
```

这样 launcher 创建 manifest 后，child worker 的 clean-worktree 检查不会把本次运行自身
误判为未提交代码。不得改用未忽略的仓库结果目录执行并发 worker。

## 固定分片与启动

- `shard0`：development 冻结索引 `[0,64)`；
- `shard1`：development 冻结索引 `[64,128)`；
- 每个 worker 始终单题 `generate`；
- 两个 worker 固定使用同一个 CUDA device 和物理 GPU UUID。

启动：

```bash
python scripts/run_cloud_v2_two_worker_generation_fixed.py \
  --training-run-dir <fixed训练校准PASS run目录>
```

断点续跑：

```bash
python scripts/run_cloud_v2_two_worker_generation_fixed.py \
  --training-run-dir <同一训练run目录> \
  --resume-run-dir <原two-worker run目录>
```

每个 shard 的已有 JSONL 必须是该 shard 冻结记录的严格前缀。发生过 worker failure 或
launcher resume 后可以继续完成完整性核对，但吞吐会标成不可直接比较。

## 合并与分析

只有两个 worker 均 PASS、adapter SHA-256 与 GPU UUID 相同、两组各 64 条完整且合并后
128 条无缺失无重复时，才创建 `merged/metrics.json` 和 `merged/manifest.json`。

与已有 single-worker batch=1 参考逐条比较：

```bash
python scripts/analyze_cloud_v2_two_worker_generation_fixed.py \
  --two-worker-run-dir <two-worker run目录> \
  --batch1-reference-run-dir <single-worker batch1 run目录> \
  --output <新的analysis.json路径>
```

PASS 要求 record 顺序、`raw_output`、`parse_status`、`parsed_prediction` 和
`numeric_correct` 全部完全一致。报告包含端到端 wall time、每个 worker 的加载/生成时间、
进程内峰值 allocated/reserved 显存和 GPU UUID。两个进程峰值之和不是板卡同时刻全局峰值。
