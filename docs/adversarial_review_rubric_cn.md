# 审核线程中文 Rubric

审核线程是只读反对线程。它的任务不是鼓励主线程，而是阻止错误方向、薄弱证据、泄漏、baseline 不公平和对老师材料的过度包装。

## 固定审核步骤

每次收到审查包后，审核线程必须按顺序做：

1. 确认只读：不得改文件、提交、推送、训练模型或改变仓库状态。
2. 核对阶段：确认 `stage_id` 是否存在于 `workflow/stages.json`，并检查该阶段是否允许推进。
3. 核对证据：至少检查 changed files、关键 artifact、验证命令三类证据，不能只复述主线程描述。审查包必须覆盖该阶段 required artifacts 和 required checks，且验证命令状态必须为 passed。
4. 攻击假设：主动寻找 test leakage、baseline unfairness、selection signal 太弱、placeholder 误用、不可复现、教授材料夸大。
5. 打分：按下方评分表逐项给分，并解释扣分原因。
6. 给 blocker：如果存在硬阻塞，必须列入“阻塞项”，并把阶段判定设为不允许进入下一阶段。
7. 给必修复：每个 major concern 至少对应一个 required fix。
8. 中文输出：除文件名、命令、模型名、字段名外，解释和结论必须使用中文。

## 审核评分表

满分 20 分。通过条件是无 blocker、总分至少 14，并且前五个核心项都至少 2 分。

| 指标 | 分值 | 必须检查 |
| --- | ---: | --- |
| test leakage / split contamination | 0-3 | dev/test 是否混用、是否有 exact/near duplicate、是否 test-driven selection |
| matched-random / baseline fairness | 0-3 | 预算、strata、overlap、token/长度、answer scale 是否可比 |
| selection signal validity | 0-3 | 是否只是 task/difficulty 重采样，是否有 error-type 或 ablation 证据 |
| result overclaiming risk | 0-3 | simulated、smoke、placeholder 是否被写成真实结果 |
| reproducibility | 0-3 | 命令、seed、raw output、model/tokenizer revision、环境是否记录 |
| professor-facing wording | 0-2 | 是否能让老师正确理解“已完成”和“未完成” |
| actionability | 0-2 | 反馈是否能直接转成修复任务 |

## 固定中文输出结构

审核线程必须输出下面结构：

```text
阻塞项:
- ...

主要问题:
- ...

次要问题:
- ...

必修复:
- ...

评分表:
- test leakage / split contamination: x/3, 原因...
- matched-random / baseline fairness: x/3, 原因...
- selection signal validity: x/3, 原因...
- result overclaiming risk: x/3, 原因...
- reproducibility: x/3, 原因...
- professor-facing wording: x/2, 原因...
- actionability: x/2, 原因...

阶段判定:
- 是否允许进入下一阶段: 是/否
- 允许的下一阶段: ...
- 禁止声明: ...
- 结论: ...
```

## 硬性 blocker

出现任一情况，默认不允许进入下一阶段：

- test split 被用于选择、调参、prompt 修改或错误分析。
- matched random 与 targeted 大量重叠，或没有记录 overlap audit。
- 用 simulated diagnostic 或 smoke test 支撑真实模型效果结论。
- 真实 diagnostic 没有 raw outputs、prompt template、decoding config、parser version、model/tokenizer revision。
- LoRA 对比没有相同预算、相同训练配置、相同 eval 集和可复现日志。
- 教授材料把 pipeline scaffold 写成已经证明有效的方法。
