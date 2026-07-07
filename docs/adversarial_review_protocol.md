# 反对线程审核协议

## 审核线程

线程标题：

```text
EG-SFT adversarial reviewer
```

第一次实现时创建的线程 id：

```text
019f3b8b-a025-7fe3-9dca-434d1e78cfa8
```

审核线程只读。它不能编辑文件、提交、推送、训练模型或改变仓库状态。

审核线程必须用中文解释和给 verdict。权威 rubric 是 `docs/adversarial_review_rubric_cn.md`，外部资料核验由 `docs/reviewer_external_evidence_policy_cn.md` 约束，结构化输出应匹配 `workflow/templates/review_response.json`。

## 审核节奏

采用阶段级审核。每个主要阶段结束后发送一次 review package，不对每个小命令审核。

这是项目固定 workflow。任何后续 model diagnostic、selector 改动、LoRA run、结果表或导师材料，都必须经过这个 gate，才可以被视为可信项目证据。

阶段包括：

1. 仓库/spec/文档更新。
2. 真实 base diagnostic。
3. B128 selection 和 bias audit。
4. LoRA smoke/full run。
5. Base/Random/Targeted comparison。
6. 导师汇报 summary。

## 审查包模板

使用 `workflow/templates/review_package.json`。发送前必须校验：

```powershell
python scripts/validate_workflow_packet.py --kind review_package --path workflow/templates/review_package.json
```

## 必须覆盖的审核 verdict

每次审核必须明确覆盖：

- test leakage risk
- matched-random fairness
- selection signal 是否超过 task/difficulty resampling
- placeholder 或 result-overclaiming risk
- 是否可以进入下一阶段

每次审核还必须包含：

- `阻塞项`
- `主要问题`
- `次要问题`
- `必修复`
- `评分表`
- `阶段判定`

## 第一轮审核摘要

第一轮审核指出过这些问题：

- Matched-random 实现和文档需要说明：为了严格 stratum matching，必要时可能与 targeted subset 重叠。
- Bias audit 文档曾声称会报告 marginal 和 mean statistics，但早期输出只报告了 stratum counts。
- Split seeds 虽然独立，但模板较窄，真实 test claim 前仍必须做 near-duplicate audit。
- 面向导师的表述必须区分 pipeline placeholder 和真实模型结果。

<details>
<summary>English note</summary>

This protocol keeps the adversarial reviewer read-only and Chinese-output-only. English research terms are retained where they are standard in the literature.

</details>
