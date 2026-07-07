# 数据生成规范

## 设计目标

生成可由 solver 验证答案的数值推理任务，并保留可控 metadata。题面可以使用商业或财务风格变量，但答案完全由程序计算，不依赖任何外部金融知识。

## 任务族

### 比例变化

给定初始值和百分比增减，计算最终值或净变化。

示例：

```text
A metric starts at 120 and increases by 15%. What is the final value?
```

### 乘法关系

给定链式乘数或单位数量，计算派生数量。

示例：

```text
Each unit contains 4 packs and each pack has 12 items. How many items are in 7 units?
```

### 加权聚合

从两到四个组件计算加权平均或加权总量。

示例：

```text
Group A has weight 0.35 and value 80, Group B has weight 0.65 and value 92. What is the weighted value?
```

### 时间数值约束

按时间顺序跟踪数值变化，包括增加、减少、上限或下限约束。

示例：

```text
A count is 50 on Monday, gains 8 on Tuesday, then loses 6 on Wednesday. What is the Wednesday value?
```

## 样本 Schema

每条样本包含：

- `id`
- `split`
- `task_family`
- `difficulty`
- `prompt`
- `answer`
- `rationale`
- `metadata`
- `buckets`

必须包含的 bucket：

- `difficulty_bucket`
- `answer_magnitude_bucket`
- `reasoning_length_bucket`

## 确定性

所有生成脚本都接受 seed，并写出可复现 JSONL。生成器只使用 Python 标准库中的 `random.Random`，避免隐式全局随机状态。

## 泄漏规则

- 不同 split 使用独立 seed 生成。
- 选择策略可以读取 candidate metadata 和 dev diagnostic 的聚合错误画像。
- 选择策略不能读取 test predictions 或 test metrics。
- OOD template 使用 held-out surface templates。
- OOD range 使用大于 candidate/dev 的数值范围。
- 真实模型比较前必须运行 duplicate/near-duplicate audit。
