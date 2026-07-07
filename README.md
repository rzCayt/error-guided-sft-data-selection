# Error-Guided SFT Data Selection

> 中文项目名：基于错误诊断的 SFT 数据选择
> 用途：港中深大模型 RA 申请项目材料与可复现实验脚手架

本仓库研究一个小而可验证的问题：**能否用 base model 在诊断集上的错误，指导 SFT 训练样本选择，并在相同样本/训练预算下优于 carefully matched random baseline？**

这个项目不是金融大模型、不是投资建议系统，也不是金融问答 benchmark。仓库中的“收益率、比例、加权、时间序列”等表述只是可控的数值推理外壳，核心目标是验证 post-training/data selection 研究流程是否严谨。

启用 GitHub Pages 后，双语项目主页通常位于 `https://rzcayt.github.io/error-guided-sft-data-selection/`，并提供中文/英文切换按钮；启用步骤见 [GitHub 设置说明](docs/github_setup.md)。仓库内的 [docs/index.html](docs/index.html) 是 Pages 源文件，如果 Pages 尚未启用，在 GitHub 上点击它只会看到源码。GitHub README 本身不能运行 JavaScript，所以 README 采用中文优先、英文折叠保留的形式。

<details>
<summary>English short description</summary>

This repository is a reproducible research scaffold for studying whether base-model diagnostic errors can guide data selection for data-efficient LoRA SFT on small numerical-reasoning language models. Finance-style variables are only used as controlled wording for solver-verifiable numerical tasks. Current result files are pipeline placeholders unless explicitly marked as real model outputs.

</details>

## 研究问题

小模型在数值推理中常见失败包括变量绑定错误、公式选择错误、尺度/单位错误、时间顺序错误和输出无法解析。本项目把这些 base diagnostic 失败转成数据选择信号：

1. 先在锁定的 `dev_diagnostic` 上评估 base model。
2. 把失败样本映射到 error taxonomy。
3. 从独立的 `candidate_pool` 中选择与弱点匹配的 SFT 样本。
4. 构造 matched-random baseline，控制任务类型、难度、答案量级和推理长度。
5. 只在锁定的 ID/OOD test split 上做最终评估。

预期贡献不是“已经证明 targeted selection 更好”，而是建立一个可审计、可复现、能防止泄漏和过度声明的实验流程。

## 当前状态

- 已实现确定性数据生成器与 solver。
- 已固定 split：`candidate_pool`、`dev_diagnostic`、`test_id`、`test_ood_template`、`test_ood_range`。
- 已实现 parser、metric、error taxonomy、base diagnostic 占位流程。
- 已实现 error-guided selector 与 matched-random selector。
- 已实现 selection bias audit 与 split leakage audit。
- 已实现 LoRA smoke 接口；无 GPU/依赖时会生成 no-training evidence，而不是伪造训练结果。
- 已建立中文主线程 workflow 与中文 adversarial reviewer 流程，要求审核线程搜索外部资料后再给阶段 verdict。

## 证据边界

当前 `results/main_results_v0.csv` 是 **simulated diagnostic placeholder**，只能用于验证文件格式、parser、selector 和评估管线是否连通。它不能作为“error-guided selection 真实提升模型”的证据。

第一个可对外讨论的真实实验必须满足：

- 使用真实模型输出替代 simulated diagnostic。
- 起点模型至少为 `Qwen/Qwen2.5-0.5B` 或同级小型开源模型。
- 保存 raw outputs、prompt template、decoding config、parser version、model/tokenizer revision、dtype、seed。
- 在相同训练预算下比较 Base、Matched Random、Targeted。
- 在比较前完成 leakage、overlap、bias audit。

## 快速开始

```powershell
cd E:\RA准备\07_error_guided_sft_repo
python -m pip install -e .[dev]
python scripts/generate_data.py --all
python scripts/audit_splits.py
python scripts/run_base_diagnostic.py
python scripts/build_selection_sets.py --budget 128
python scripts/evaluate_results.py
pytest -q
```

可选训练/模型 smoke test：

```powershell
python -m pip install -e .[train]
python scripts/run_qwen_smoke.py
python scripts/run_lora_smoke.py --model Qwen/Qwen2.5-0.5B
```

## 仓库结构

- `docs/project_spec.md`：中文研究设计、split 纪律、评估计划。
- `docs/data_generation_spec.md`：中文数据生成规范、任务族、solver 保证。
- `docs/selection_policy_spec.md`：中文 targeted/matched-random 选择策略。
- `docs/literature_review.md`：数据选择与 LoRA 相关文献定位。
- `docs/contribution_statement.md`：和老师说明“我做了什么”的中文表述。
- `docs/professor_summary_1p.md`：一页中文导师汇报摘要。
- `docs/workflow_cn.md`：中文主线程 SOP 与评分门槛。
- `docs/adversarial_review_rubric_cn.md`：中文审核线程 rubric。
- `docs/reviewer_external_evidence_policy_cn.md`：审核线程外部资料搜索规则。
- `docs/index.html`：可切换中英文的 GitHub Pages 项目主页。
- `workflow`：阶段门槛、外部来源种子、审查包模板和校验 fixture。
- `src/eg_sft/data`：生成器、schema、solver。
- `src/eg_sft/eval`：parser、metric、error taxonomy。
- `src/eg_sft/selection`：error-guided selection、matched random、bias audit。
- `scripts`：可复现实验入口。

## 固定工作流

本项目后续按阶段推进：

1. 主线程生成阶段计划。
2. 实施本阶段任务。
3. 主线程填写自评。
4. 生成结构化 review package。
5. 本地校验 review package。
6. 发给只读中文审核线程。
7. 审核线程必须搜索资料、检查证据、给 blocker/verdict。
8. 修复 blocker 后再进入下一阶段。

当前允许进入的下一阶段是 `real_base_diagnostic`：用真实 `Qwen/Qwen2.5-0.5B` 或同级模型跑 base diagnostic。当前仍禁止宣称 Targeted 优于 Random。

## 质量规则

- 不用 test split 调 prompt、parser、selection policy 或训练策略。
- 不混用 `dev_diagnostic` 和 test split。
- 不把 simulated/smoke/no-training placeholder 写成真实模型结论。
- 不提交模型权重、adapter、大型生成文件或 token。
- 失败训练也要记录环境、命令、错误和下一步，而不是删除失败证据。
