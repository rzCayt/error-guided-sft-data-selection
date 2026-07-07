# 项目固定工作流

本仓库使用 stage-gated workflow。目标是让研究过程保持可信，避免隐性泄漏、弱 baseline 和过度声明。

权威流程以中文、机器可校验版本为准：

- `docs/workflow_cn.md`：主线程 SOP 与自评 rubric。
- `docs/adversarial_review_rubric_cn.md`：只读审核线程 rubric 和中文输出格式。
- `docs/reviewer_external_evidence_policy_cn.md`：审核线程外部资料搜索和引用规则。
- `workflow/stages.json`：阶段门槛、必需产物、必跑检查、禁止声明和退出条件。
- `workflow/templates/*.json`：阶段计划、自评、审查包和审核回复模板。
- `scripts/validate_workflow_packet.py`：本地 workflow packet 校验器。

## 阶段门槛

每个阶段必须按同一顺序推进：

1. 填写结构化 stage plan。
2. 执行或更新本阶段 artifact。
3. 填写主线程 self-check。
4. 生成结构化 review package。
5. 用 `scripts/validate_workflow_packet.py` 校验。
6. 把 package 发给 adversarial reviewer 线程。
7. 先修复 blocker，再进入下一 claim-bearing 阶段。
8. 只有本地验证和审核都通过后，才提交和推送。

## 审核线程

线程标题：

```text
EG-SFT adversarial reviewer
```

线程 id：

```text
019f3b8b-a025-7fe3-9dca-434d1e78cfa8
```

审核线程只读，必须中文输出，并重点攻击：

- test leakage 和 near-duplicate 风险。
- matched-random baseline 是否公平。
- selection signal 是否超过 task/difficulty 重采样。
- simulated result 是否被过度包装。
- LoRA 实验是否可复现。
- 对外研究表述是否夸大。

## 审查包要求

使用 `workflow/templates/review_package.json` 的结构。审核回复必须遵守 `workflow/templates/review_response.json` 和 `docs/adversarial_review_rubric_cn.md`。

## 当前阶段状态

允许的下一步：

- `parser_error_audit`：复核真实 Qwen base diagnostic 的错误样例，区分模型推理错误、parser/格式风险和 prompt/题目理解风险。

暂不允许：

- 有 claim-bearing 含义的 Random vs Targeted LoRA comparison。
- 对外真实性能提升声明。

进入 LoRA 对比前必须完成：

- 运行 split leakage audit。
- 对准确预算运行 selection bias audit 和 strong baseline audit。
- 要求 `overlap_rate=0`，或明确把 comparison 标记为有独立性限制。
- 至少完成 exact matched random multi-seed、stratified random 和 metadata-hard baseline。
- 加入或消融 error-type-aware selector。
- 用真实模型 raw outputs 和 run metadata 替换 simulated diagnostic rows。
- 完成真实错误样例和 parser 风险复核，避免把格式问题当作推理提升空间。
- 构建选择集时显式使用 `--profile results/real_error_profile.csv`，不能继续使用 simulated `error_profile_v0.csv`。
- 把 Qwen smoke evidence 保存为机器可读 artifact，而不是只写 prose。

<details>
<summary>English note</summary>

This file is Chinese-first while retaining key English research terms to keep the project searchable and readable on GitHub.

</details>
