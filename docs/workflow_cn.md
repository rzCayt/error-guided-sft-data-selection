# 主线程固定工作流

本项目以后按阶段推进。主线程不能只凭感觉继续加功能，也不能在没有证据 artifact 的情况下把阶段说成完成。

## 固定顺序

每个阶段必须按下面顺序执行：

1. 写阶段计划：填写 `workflow/templates/stage_plan.json` 的同结构文件，明确阶段、目标、产物、验证命令、禁止声明和停止条件。
2. 执行阶段任务：只做本阶段允许的改动，不顺手扩展无关功能。
3. 主线程自评：填写 `workflow/templates/main_self_check.json` 的同结构文件，按评分表打分。
4. 生成审查包：填写 `workflow/templates/review_package.json` 的同结构文件，列出证据、命令、结果、弱点和要求审核线程回答的问题。
5. 自动校验审查包：运行 `scripts/validate_workflow_packet.py`，校验结构、证据路径、验证命令和中文要求。审查包必须覆盖该阶段在 `workflow/stages.json` 中列出的 required artifacts 和 required checks。
6. 发送给审核线程：审核线程必须只读、中文输出、按 rubric 审查，并按 `docs/reviewer_external_evidence_policy_cn.md` 搜索外部资料。
7. 修复 blocker：如果审核线程给出 blocker，主线程必须先修复并重新发审查包，不能进入下一阶段。
8. 提交和推送：只有本地验证通过、审核通过、工作树范围明确时才提交推送。
9. 记录下一阶段：在汇报中明确下一阶段 id、允许做什么、禁止声称什么。

## 主线程自评评分

满分 19 分，通过条件是无硬 blocker、总分至少 16，并且“可复现性、泄漏安全、声明克制程度”都至少 2 分。

| 指标 | 分值 | 低分含义 |
| --- | ---: | --- |
| 目标贴合度 | 0-2 | 做了和当前阶段无关的事 |
| 证据完整性 | 0-3 | 没有 artifact、日志、结果表或代码证据 |
| 可复现性 | 0-3 | 缺少命令、seed、环境、模型 revision 或 raw output |
| 泄漏安全 | 0-3 | 可能使用 test 信息调参或选择数据 |
| baseline 公平性 | 0-3 | Targeted/Random 不可比，或预算、strata、overlap 未审计 |
| 声明克制程度 | 0-3 | 把 smoke、placeholder、simulated 结果说成真实结论 |
| 下一步清晰度 | 0-2 | 没有说明下一阶段入口和停止条件 |

## 无意义推进判定

出现任一情况，主线程必须停止并重写计划或补证据：

- 没有新 artifact，却声称阶段完成。
- 必跑验证命令失败或未运行，却把审查包标记为可通过。
- 审核线程没有外部资料核验，却允许进入真实模型、selection 或 LoRA 阶段。
- 用 simulated placeholder 支撑真实模型结论。
- 没有 overlap、bias、leakage audit 和 strong baseline audit，就比较 Targeted 和 Random。
- 没有 raw outputs 和 run metadata，就宣称真实 diagnostic。
- 只改展示材料、不增加证据，却推进到 external-facing claim。
- 重复运行同一命令，但没有新假设、新失败分析或新修复。
- 审核线程指出 blocker 后，没有修复就继续下一阶段。

## 当前允许方向

当前允许进入的下一阶段是 `real_base_diagnostic`，只限探索性真实 base diagnostic。仍然禁止声明 Targeted selection 优于 Random，也禁止把 simulated diagnostic 当成真实模型证据。
