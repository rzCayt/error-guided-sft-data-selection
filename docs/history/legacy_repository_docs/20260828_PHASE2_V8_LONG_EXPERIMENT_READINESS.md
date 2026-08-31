> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# Phase 2 v8 长实验准备状态与执行协议

日期：2026-08-28  
状态：CPU 工程准备完成；GPU 正式矩阵尚未获准启动。

## 一、这轮实验到底回答什么

主问题不是“RDS 一定比随机抽样好”，而是：

> 在样本数、回答监督词元、来源和回答长度组成相同后，RDS-error 的名单排序是否比随机排序带来稳定的下游收益？

本轮只运行可识别度最高的 common-mix 比较：

- `random_common_mix`；
- `rds_error_common_mix`；
- 4 份冻结名单 replicate（不假设彼此独立，RDS 重合度单独报告）；
- 训练种子 17、29、41；
- 共 24 个全新 cell。

旧环境的 8 个 common seed17 cell 仅作为历史外部复现证据，不进入新 24 格主分析。free-mix、第二模型、第二预算和新 selector 均不属于本轮。

## 二、为什么改成 24 个全新 cell

原方案把历史 seed17 与新机器上的 seed29/41 合并，会把训练随机种子、硬件、驱动、代码时期和执行时间混在一起。v8 在同一套代码和同型号 GPU 上重新运行三个种子，主比较才是干净的交叉设计。

## 三、已经完成的 CPU 证据

- 唯一正式矩阵：`configs/phase2_clean_common24_v8_canonical.json`；
- 唯一运行权威清单：`configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json`；
- 24/24 resolved contract 字段级比较；
- 24/24 真实 Qwen2 tokenizer 输入、label mask、样本顺序、优化步和随机数映射哈希；
- 同一名单跨种子时，选样、token、标签和训练配置完全相同；
- 由种子决定的样本 occurrence、step plan 和 RNG 哈希均按预期不同；
- 主统计固定观察到的三个训练 seed block；随机与 RDS 名单分别重采样，题目在方法间共享；对 seed 总体的重采样仅作探索性敏感性分析；
- 1pp 结论规则已纠正：只有 95% CI 下界大于 +1pp 才能称为“至少 1pp 的有意义提升”；
- 精度模拟已将 ±1pp 等效结论降级为探索性；
- 每张物理 GPU 同一时刻只运行一个训练或推理进程；
- 每 worker 独立状态目录、attempt 和证据包；
- 锁、崩溃恢复、重复启动、原子成包和 COMPLETE 终态已有失败注入测试。

## 四、GPU 选择

正式方案使用两台独立的单卡 RTX 4090D 实例，每台 12 格；不使用 DDP，也不租同一台双卡主机。

原因：

1. 两台独立实例避免共享 CPU、内存和数据盘产生竞争；
2. 两张卡分配相同数量的方法、名单和种子，避免方法绑定到某张物理卡；
3. 任何一台中断时，另一台仍可继续；
4. 4090D 已足够承载 1.5B LoRA，5090/PRO 6000 不能提高本轮结论的可信度。

本 release 不支持单卡冒充双 worker 顺序运行。如果只能获得一张卡，必须另建并重新审计 single-worker 协议；当前正式块不启动。

## 五、开卡后的强制顺序

### Q0：主机 CPU 预检

每台实例先运行 `scripts/phase2_v8_prepare_host.sh`，核验部署包、模型快照、24 格输入合同、磁盘，并在 `HF_DATASETS_OFFLINE=1` 下逐条验证 GSM8K、SVAMP、ASDiv 和 MultiArith 的 pinned cache。该脚本不训练模型。

### Q1：推理环境桥接

1. GPU0 跑 base16 和历史 adapter128，生成本轮 token anchor；
2. GPU1 对同一批样本逐 token 匹配 GPU0；
3. 两台机器的 Python、PyTorch、CUDA、Transformers、PEFT、驱动、模型文件和数值设置必须一致；
4. 仍固定 batch size 1，不重新尝试 batch>1。

### Q2：训练路径桥接

GPU0 用 `v8_rep1_random_common_mix_train17` 完整运行 fresh A1 和 fresh-process A2；GPU1 运行 B1，然后比较：

- 真实输入合同完全一致；
- 每步监督 token 数完全一致；
- loss 轨迹最大差异不超过 `1e-5`；
- adapter 最大逐元素差异不超过 `1e-6`；
- adapter 余弦相似度不低于 `0.9999999`；
- 128 题逐 token 输出一致。

任何一项失败，24 格矩阵不启动，也不扩大阈值强行通过。

### Q3：24 格长期运行

- GPU0、GPU1 各 12 格；
- 每格按固定顺序完成：训练 → adapter 保存/重载 → GSM8K → SVAMP → ASDiv numeric → MultiArith → 正式审计 → OOD 审计 → 证据包；
- 一个 cell 全部通过后，盲态 supervisor 自动开始本 worker 的下一格；
- 不读取方法准确率，不自适应改参数；
- 24/24 `AUDITED_PASS` 前不统一解盲。

## 六、停止和恢复规则

- GPU 达到 80°C：当前进程安全停止并保留断点；
- SSH 中断不等于任务失败，先检查进程、run manifest 和已有 shard；
- 恢复只能写入原 attempt 或创建有证据关联的新 attempt；
- 不复用不完整 adapter；
- 不覆盖已有 raw output、audit 或 evidence package；
- 每张 GPU 同时只允许一个 GPU 进程，CPU 哈希与打包可并行；
- 任何 canonical 文件或 SHA 改变，都必须重新生成版本化清单并重新通过 CPU 闸门。

## 七、预计时间与费用

以历史完整 cell 约 3 GPU 小时估算：

- 24 格约 72 GPU 小时；
- 两张独立 4090D 的理想墙钟约 36 小时；
- 加上 qualification、数据传输、中断恢复和审计，保守估计 40–52 小时；

实际费用按两台实例的实时单价计算。达到预设预算上限必须暂停，不自动扩展到 free-mix、第二预算或第二模型。

## 八、本轮能够支持和不能支持的结论

能够支持：

- 当前 Qwen2.5-1.5B、B=500、固定 common-mix 设定中 RDS-error 排序相对随机排序的方向和不确定性；
- 名单差异、训练种子差异和测试题差异对结果的相对贡献；
- 当前配置是否出现稳定方向、实际意义或仅仅证据不足。

不能支持：

- RDS 在所有模型、预算和任务上有效或无效；
- error-conditioning 相对 all-query 有新增信号；
- free-mix 的来源/长度机制；
- ±1pp 等效，除非 90% CI 实际完全落入该区间；
- 顶会级一般性结论。

## 九、真正开始 GPU 前仍需完成

1. 最终全量 CPU 测试和 fresh-extract 测试通过；
2. 生成唯一 v8 部署包及包级 SHA-256；
3. 将 v8 源码形成一个可审计的干净 Git 提交并部署为 clean clone；
4. 用户提供两台独立 4090D 当前 SSH 信息；
5. 先只运行 Q0–Q2；只有训练锚点 PASS 后生成 `READY_FOR_HUMAN_REVIEW`；只有人工发送 `START_PHASE2_V8_COMMON24` 才进入 24 格。
