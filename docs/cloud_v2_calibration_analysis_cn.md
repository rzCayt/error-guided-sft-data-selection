# Cloud-v2 校准结果分析说明

## 训练校准比较

四个 64 条训练校准任务全部完成后运行：

```bash
python scripts/analyze_cloud_v2_training_calibration.py \
  --run mb1_ga16=<run目录> \
  --run mb2_ga8=<run目录> \
  --run mb4_ga4=<run目录> \
  --run mb8_ga2=<run目录> \
  --output <新的analysis.json路径>
```

分析器会验证 profile、协议/配方/选样哈希、optimizer steps、response token 数、
温度采样次数和 adapter 重载结果。参数方向比较使用：

```text
真实更新向量 = 最终 checkpoint adapter_state - 初始 checkpoint adapter_state
```

不得用最终 adapter 参数本身的 cosine 代替更新向量 cosine。

当前 checkpoint 没有保存每一步原始梯度，因此报告中的
`gradient_history_proxy_cosine_vs_mb1` 来自最终 AdamW `exp_avg`。它只是梯度历史代理，
不能表述为 raw-gradient cosine。

## 生成校准比较

四个 128 条 development 生成任务完成后运行：

```bash
python scripts/analyze_cloud_v2_generation_calibration.py \
  --run b1=<run目录> \
  --run b4=<run目录> \
  --run b8=<run目录> \
  --run b16=<run目录> \
  --output <新的analysis.json路径>
```

分析器逐条比较：

- `record_id` 顺序；
- `raw_output`；
- `parse_status`；
- `parsed_prediction`；
- `numeric_correct`。

吞吐主指标为完整 128 条的 `wall_examples_per_second`。当前生成入口记录的
`generated_token_count_this_invocation` 可能包含同一 batch 内 EOS 后补齐的位置，因此
在没有明确的 unpadded token count 时，分析器不会比较 token/s。

## 结论边界

这两个工具只判断 cloud-v2 执行配置是否可比、是否更快，不评价 selector 是否有效，
也不产生 held-out GSM8K 结论。只有校准分析通过并冻结新的正式配置后，才可创建新的
九格确认性矩阵。
