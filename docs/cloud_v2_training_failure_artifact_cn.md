# Cloud-v2 训练校准失败记录规范

当某个 profile（例如 `mb8_ga2`）发生真实 CUDA OOM 时，不得伪造 checkpoint、update
vector 或 PASS metrics。应根据真实日志手工创建一个新的、不可覆盖 failure JSON，再交给
分析器读取。

下面只是字段模板，不是实验结果：

```json
{
  "failure_schema_version": "cloud-v2-training-calibration-failure-v1",
  "profile": "mb8_ga2",
  "status": "FAIL",
  "failure_kind": "cuda_out_of_memory",
  "stage": "填写真实失败阶段",
  "exception": {
    "type": "填写真实异常类型",
    "message": "填写真实异常消息"
  },
  "gpu": {
    "uuid": "填写真实GPU UUID",
    "name": "填写真实GPU名称",
    "total_memory_gib": 0,
    "peak_allocated_memory_gib": 0,
    "peak_reserved_memory_gib": 0
  },
  "input_contract": {
    "calibration_config_hash": "来自真实run contract",
    "protocol_config_sha256": "来自真实run contract",
    "base_recipe_config_sha256": "来自真实run contract",
    "selection_manifest_sha256": "来自真实run contract",
    "selected_id_sha256": "来自真实run contract"
  },
  "source_log_sha256": "真实失败日志的64位SHA-256",
  "recorded_at_utc": "真实UTC时间"
}
```

三个成功 run 加一个真实 OOM failure 的分析命令：

```bash
python scripts/analyze_cloud_v2_training_calibration_with_failures.py \
  --run mb1_ga16=<PASS run目录> \
  --run mb2_ga8=<PASS run目录> \
  --run mb4_ga4=<PASS run目录> \
  --failure mb8_ga2=<不可覆盖failure.json> \
  --output <新的analysis.json路径>
```

失败档会直接淘汰，不参与 update cosine。成功档仍必须从各自初始和最终 checkpoint 计算
真实 `final - initial` update vector。分析器不会自行生成 failure JSON，也不会补写缺失显存值。
