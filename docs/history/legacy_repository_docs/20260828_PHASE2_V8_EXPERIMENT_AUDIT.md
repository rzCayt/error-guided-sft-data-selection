> **Historical snapshot.**
>
> This document records an earlier research stage and is not the current result.
>
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# Phase 2 v8 对抗审计整改记录

外部审计输入：用户提供的 2026-08-28 GPT Pro 对抗审计包，SHA-256 为 `73600259cd500cc2aa716c3479259a9adf89642f50e5713e7f560731db53687c`。

## 审计裁决

原 v7 方案：不允许直接运行。  
修订后的 v8：CPU 工程闸门已通过；仍需双 GPU 推理桥接和训练锚点通过，才能运行正式 24 格。

## P0 整改映射

| 问题 | v8 处理 | 当前状态 |
|---|---|---|
| 审核包没有真实源码 | 部署包和审核包都纳入真实源码、测试与输入合同 | 待最终成包复验 |
| 多份冲突配置 | 唯一 canonical matrix + 运行文件 SHA 清单，worker 拒绝非权威文件 | 已实现 |
| 历史 seed17 与新 seed 混杂 | 三个 seed 全部在新环境重跑；历史 cell 仅作外部证据 | 已冻结 |
| “只改变 seed”只有声明 | 物化 24 格 resolved diff、token、label、顺序、step 和 RNG 哈希 | 24/24 PASS |
| bootstrap 破坏交叉结构 | seed 全局共同重采样；两种方法名单分别重采样；题目共享 | 已实现 |
| GLMM 缺少 cell 相关 | 稳健性模型加入 cell 级结构；小样本方差分量只作探索性 | 已冻结 |
| 1pp 规则过强 | 有意义提升要求 95% CI 下界 > +1pp；等效要求 90% CI 完全在 ±1pp | 已修复 |
| canary/hash 语义不清 | 固定 prompt、attention、raw continuation、首 EOS、decoded、parser、correctness、strict 多层签名 | 已实现，待 GPU |
| 双机 registry 竞态 | 每 worker 独立状态目录和证据包；单 GPU 进程；失败注入测试 | 已实现 |
| free-mix 占用低识别度预算 | 正式块缩减为 24 个 clean common cell | 已冻结 |

## P1 整改

- seed、replicate、方法在两张卡和运行波次中均衡轮换；
- RDS 高名单重合只作为敏感性信息，不当作四个完全独立政策；
- 研究称为 prospective clean replication，不声称原始完全确认实验；
- OOD 不建立未预注册的硬裁决，逐任务描述并保留最差任务；
- 使用两台独立单卡实例，避免同主机资源竞争；
- 盲态 supervisor 可自动续跑，但不能读取准确率或改参数。

## v8.2 新增整改

- Canonical authority 固定为 `CANONICAL_RUNTIME_FILES_v8_RELEASE.json`，Python 入口不能替换为旧 manifest；
- Training-anchor cross/same drift ratio 降级为诊断项，冻结绝对门槛与 128 题 exact signature 为权威闸门；
- 四个正式评估数据集新增离线 cache qualification，contract SHA 写入 environment；
- ±1pp 等效不再输出为主结论，只保留探索性信号；
- 正式 worker shell 在 Python 启动前冻结 `PYTHONHASHSEED`、CUBLAS 与 offline 环境；
- environment contract 补齐 tokenizers、safetensors、NumPy、PyArrow、HF Hub、fsspec、dill 和 multiprocess 的精确版本；
- preflight、training anchor 和正式 worker 均把 materialized input root 绑定到 canonical role，不能换成另一份 PASS 目录。

## 当前剩余阻塞

- 尚未在两张真实 4090D 上获得 environment equality；
- 尚未完成 base16、adapter128 的新环境逐 token 桥接；
- 尚未完成两张卡的 seed17 训练锚点；
- 尚未形成用户授权的干净 Git 提交与远端 clean clone。

因此当前结论是：`CPU_READY / GPU_QUALIFICATION_REQUIRED / FORMAL_MATRIX_NOT_AUTHORIZED`。
