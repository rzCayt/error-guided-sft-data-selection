> **Historical snapshot.**
>
> This document records an earlier research stage and is not the current result.
>
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# random-500 LoRA 工程闭环记录

更新日期：2026-07-29

## 已完成的内容

本阶段只验证一件事：固定 `random` 选样、`B=500`、训练种子 17
时，LoRA 是否能够完成训练、保存、重新加载和完整 GSM8K 评估。

训练进程保留下来的原始 stdout 显示：

- 第 1 轮 token loss：0.943357；
- 第 2 轮 token loss：0.889213；
- 日志中的轮末 optimizer step 分别为 31 和 62；
- 原单体进程在评估到 8/1,319 后中断，但 adapter 已经保存。

随后使用不改变题目、prompt、parser、生成参数和 adapter 的可恢复温控评估，
经过 v1、v2、v3 三段运行完成全部 1,319 道 GSM8K 测试题。

## 最终评估结果

| 指标 | 结果 |
|---|---:|
| 测试题数量 | 1,319 |
| 数值答对 | 808 |
| 数值准确率 | 61.2585% |
| 成功提取数值 | 1,319 |
| 数值解析成功率 | 100% |
| 严格 `Final answer:` 格式成功 | 646 |
| 严格格式成功率 | 48.9765% |
| 使用“最后一个数值”回退解析 | 673 |

“解析成功率 100%”不等于“模型答对率 100%”。它只表示每条回答都能提取出
一个数值；真正的数值准确率是 61.2585%。

## adapter 重载证据

独立的新进程重新加载了固定 base model 和磁盘中的 LoRA adapter：

- active adapter：`default`；
- LoRA 张量：392 个；
- LoRA 参数：18,464,768 个；
- 所有保存的 LoRA 参数均非零；
- 同一固定 prompt 下，启用和禁用 adapter 的 logits 最大绝对差为 2.3125。

因此可以确认 adapter 不只是文件存在，而是被成功加载并实际参与了模型计算。

## 完整性与来源

- 1,319 条输出的 ID 唯一且与冻结测试集顺序完全一致；
- 每条问题和标准答案的规范化哈希都与固定 revision 的 GSM8K 一致；
- 1,319 条结果全部重新解析、重新判分，和保存结果逐条相同；
- raw output SHA-256：
  `b8ac83f17b868f9a3fd0c7ffda132a0ae7847909c5385538fa38d6d890230d21`；
- adapter SHA-256：
  `a4231f7124ab5b225698218e43660c0083be4545ebe19d60fe7092ef426701fa`。

v1、v2、v3 的前缀行数和哈希形成连续链：

- v1：0 → 20 条；
- v2：继承 v1 的 20 条及其哈希，继续到 48 条；
- v3：继承 v2 的 48 条及其哈希，继续到 1,319 条。

原训练日志和这条恢复链保存在运行目录的 `provenance/` 中。原单体进程没有
保存的逐步训练遥测不会被事后补造。

## tokenizer 警告

Transformers 4.57.2 会提示 `fix_mistral_regex=True`。在 1,319 个固定评估
prompt 上比较后，3 个 prompt 的 tokenization 会因该开关改变。

当前 protocol v1 必须继续使用训练和本次评估共同采用的默认 tokenizer 行为，
否则属于看见结果后改变接口。未来如要采用修复，应新建 protocol v2，并在任何
训练或结果出现前冻结。

## 目前允许和不允许的结论

允许：

> random-500 的 LoRA 训练、adapter 保存、重新加载和 1,319 条 GSM8K 评估
> 工程闭环已经完成。

不允许：

> random、rds_all、rds_error 已经完成比较。

当前只有 `random`、seed 17 的完整训练结果。三策略、三随机种子的统一入口只有
在本轮完整性问题处理并复审后才准备；正式比较结果尚不存在。

## 独立复审结论

2026-07-29 的第二轮隔离复审没有发现标签、评分、行数、恢复链、文件哈希或
adapter 张量的机械错误，因此机械完整性裁定为 `PASS`。

工程闭环整体仍标记为 `WARN`，原因不是结果错误，而是 adapter/tokenizer 尚未进入
不可变外部归档、原训练进程未保存完整逐步遥测，以及 tokenizer 的跨版本行为需要
单独冻结。详细裁定见同一评估目录中的 `EXPERIMENT_REAUDIT.md`。
