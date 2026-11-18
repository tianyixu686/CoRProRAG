# cor_simple held-out 469 错误分类与问题分解审计

> 分析对象：`cor_simple_gpt4omini_plan_heldout469_0821_fixkw2_16h`。
> Planner 为 `gpt-4o-mini`，执行/阅读/最终回答为统一的本地 Qwen3-VL。
> 本报告分析的是未启用 Logic Harness 的 legacy `cor_simple` 单答案传播版本。

## 1. 原始结果

运行日志记录：

| 题数 | EM | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 469 | 34.97 | 40.82 | 39.83 | 45.38 |

按该次运行使用的原始 EM 归一化，共 164 题 exact match，305 题非 exact match。

不同标注模态路径的 EM 情况：

| 标注路径 | 题数 | EM 正确 | EM 错误 | 错误率 |
|---|---:|---:|---:|---:|
| `image→textual` | 336 | 127 | 209 | 62.20% |
| `textual→image` | 131 | 37 | 94 | 71.76% |
| `uncertain` | 2 | 0 | 2 | 100.00% |

`textual→image` 明显更难，说明系统在“先得到文本候选，再通过图像给出最终视觉答案”上尤其
不稳定。但这只是相关性：两类问题本身难度也可能不同。

## 2. 分类方法

对每个 EM 错例按 trace 中第一个可观察故障做互斥分类，检查顺序为：

1. 明确的评测归一化等价；
2. Planner 结构错误；
3. unknown 是否进入后续 resolved question；
4. 最终答案是否包含 gold 但输出过长；
5. gold 是否已在某个子问题输出中出现但后来丢失；
6. 标注 supporting image 是否进入执行 trace；
7. gold 中间答案是否进入证据和子问题输出；
8. gold 最终答案是否只存在于 evidence 中但未被抽取。

因此本表是 first-observable-failure taxonomy，不表示一个问题只存在一种缺陷。例如 Vandit
Bhatt 题同时存在 Planner 顺序错误和图像漏召回，主分类记录为图像漏召回，另用
`planner_semantic_issue_suspected` 标记 Planner 问题。

逐题审计表位于：

```text
Output/runs/multimodalqa/corprorag_fullkg_sanity/
  cor_simple_gpt4omini_plan_heldout469_0821_fixkw2_16h/
  Analysis/failure_audit.csv
```

## 3. 305 个 EM 错例的完整分类

| 首个可观察故障 | 数量 | 占 305 错例 | 含义 |
|---|---:|---:|---|
| unknown 单答案传播 | 90 | 29.51% | 前一步 `I don't know` 被写入后续问题 |
| gold 子问题输出被后续链路丢失 | 56 | 18.36% | gold 已被某一步答出，但没有成为 final |
| evidence 有 gold 但未抽取/选择 | 41 | 13.44% | reader 或 selector 没有利用已召回证据 |
| 答案过度生成/集合不精确 | 41 | 13.44% | 包含 gold，但附加了内容或多答案集合不完整 |
| gold supporting image 未召回 | 26 | 8.52% | 标注支持图像没有进入 execution trace |
| gold 中间候选在证据中但未选中 | 21 | 6.89% | candidate selection 丢失正确中间实体 |
| 中间实体正确但下游检索/推理错误 | 19 | 6.23% | 后一跳没有得到最终 gold |
| 数字词归一化错误 | 9 | 2.95% | 如 `2` 与 `two`，语义上正确 |
| 单复数归一化错误 | 1 | 0.33% | `Horse` 与 `horses`，语义上正确 |
| Planner 明确结构性欠分解 | 1 | 0.33% | 视觉多模态题只保留一个子问题 |
| **合计** | **305** | **100%** |  |

其中 10 题是明确的旧 evaluator 归一化问题。仅修正这 10 题，语义等价的 exact correct 至少
从 164 增加到 174；这不应被报告成方法提升，而应作为评测修复单独说明。

## 4. 最大问题：unknown 单答案传播

这是数量最多且最确定的实现问题。469 题中共有 109 题出现 unknown 被写入后续 resolved
question，其中 90 题最终错误。

典型例子：

```text
Q: What is the person in the S. Cole TV title cover where she played Muriel doing?
```

Planner 给出的步骤是正确的：

```text
1. What TV title did S. Cole play Muriel in?
2. In the cover image of {answer_1}, what is the person doing?
```

第一步内部的 `text_answer` 已经是 `Talking Heads`，图像候选也有
`Talking Heads (series)`，但子问题融合回答器输出 `I don't know`。legacy 执行器随后生成：

```text
In the cover image of I don't know, what is the person doing?
```

第二步自然检索到错误图像并继续失败。这不是 Planner 没有分清模态，而是执行器把一个
局部失败当成唯一状态传播。Logic Harness 保留多个候选和 dependency recovery，首先就是为
解决这一类问题。

## 5. 问题分解是否已经做好

### 5.1 结构层面基本做好

469 个 plan 的结构统计：

| 检查项 | 结果 |
|---|---:|
| 成功生成规范 plan | 469/469 |
| 出现 typed operation/modality 等额外字段 | 0 |
| 非法或向后引用的 `{answer_i}` | 0 |
| 含显式依赖关系 | 465/469 |
| 只有一个子问题 | 3/469 |

子问题数量分布为：2 个节点 222 题、3 个节点 163 题、4 个节点 52 题、5 个节点 24 题、
6 个节点 5 题、1 个节点 3 题。

所以从 JSON 合法性、依赖关系和“只分自然语言问题、不输出 operation”来看，GPT Planner 已经
相当稳定。不能再把 305 个错误笼统归结为问题分解失败。

### 5.2 语义顺序仍未完全做好

结构正确不等于检索顺序正确。在 26 个 gold image miss 中逐项检查，至少 15 个存在明确的
anchor-first 问题：Planner 先全库搜索泛化视觉特征，没有先利用题目中的人物、作品、联赛或
候选列表缩小范围。

典型例子一，Vandit Bhatt：

```text
错误计划：先找所有 poster 有白色穹顶建筑的电视剧
正确计划：先找 Vandit Bhatt 的电视剧候选，再逐个检查 poster
```

错误计划检索出 `Silver Spoon, Sterling Shackles` 和 `River of Wine`，gold
`Madam Secretary` 的支持图像没有进入 trace，最后角色被答成 `Chung Ho-yee`。

典型例子二，Anna Akana：

```text
错误计划：先找 title screen 显示房子的所有 TV show
正确计划：先找 Anna Akana 出演的 TV show，再检查候选 title screen
```

全库视觉检索选成 `The Waltons`，后续自然无法得到角色 `Lily`。

典型例子三，Ranjan Pramod：Planner 虽然建立了人物电影分支，但“poster 只有两个女人”的
视觉分支没有依赖该候选集合，结果全库搜到 `Practical Magic`，两条分支在比较前已经脱节。

其他明确问题包括：

- 从所有 2016 TV shows 开始，而不是先找 David Buckley 参与的 shows；
- 从所有 sword poster 开始，而不是先找 Robertryan Cory 参与的 films；
- 从所有 bull-logo clubs 开始，而不是先找 Facundo Lescano 生涯中的 clubs；
- 漏掉“bald head”视觉验证，直接把 top assist player 当成目标人物。

因此准确结论是：**Planner 的结构生成已经足够稳定，但候选集合优先、视觉条件后置的语义
规划仍需加强。** 目前没有人工标注全部 469 个 plan 的 gold decomposition，15 只是已确认的
下界，不能当作全部 Planner 语义错误数量。

## 6. 检索之后仍不会用证据

### 6.1 Gold 已在 step output 中却被丢掉：56 题

例如 Guapore 比较题，前序步骤已经包含 `Guaporé`，但另一分支输出 unknown，final 阶段没有
基于有效分支完成比较，最后变成 `I don't know`。

这说明旧版只有“子问题答案列表”，没有全局候选路径和约束一致性。一个正确局部答案不会被
保护，也无法与其他分支形成可回溯比较。

### 6.2 Gold 只在 evidence 中，没有被抽取：41 题

例如 thrush 题的计划是合理的：

```text
Wales 鸟类候选 -> 看眼部条纹 -> 查询 binomial
```

正确中间候选 `Redwing` 已在第一跳列表和图像候选中，但视觉解释把
`Eyebrowed thrush/Dusky thrush` 也判断为满足条纹，最终选择 `Turdus obscurus`，而不是
`Turdus iliacus`。

这里主要是细粒度视觉验证和 selector 问题，不是分解错误。

### 6.3 中间候选在 evidence 中但未被选中：21 题

这类题的正确实体已经被召回，但单个 sub-answer 选择了另一个高分候选。旧版只传播 top-1，
因此后续跳无法恢复正确实体。

### 6.4 中间实体正确，但下一跳仍失败：19 题

例如 Reba McEntire 题，第一步正确找到演唱会 headliner `Reba McEntire`，第二步查询
`Who sang Does He Love You with Reba McEntire?` 时却答成 `Loretta Lynn`，gold 是
`Linda Kaye Davis`。这里应归因于下游文本检索/阅读，而不是 Planner 或图像检索。

## 7. 输出与评测问题

### 7.1 过度生成：41 题

例如 gold 为：

```text
Hero Girl (voice)
```

预测为：

```text
Hero Girl (voice) and motion-capture
```

核心实体和角色已经找到，但 final 没有遵守短答案契约，EM 记为 0。类似地，`right hand`
相对 gold `right`、`a white goalkeeper glove` 相对 gold `glove` 都属于这一类。

其中也有多答案集合不完整或混入额外候选，不能全部视为纯格式错误。因此应同时报告 F1，
并单独统计 exact-span formatting error。

### 7.2 明确归一化误判：10 题

9 题为数字与数字词，例如：

```text
prediction = 2, gold = two
```

另有：

```text
prediction = Horse, gold = horses
```

这些不属于方法错误。数字词问题已能通过新版 normalization 修正，单复数是否归一化需要和
官方 MMQA evaluator 规则保持一致后再决定。

## 8. 为什么 textual→image 更差

`textual→image` 错误率为 71.76%，高于 `image→textual` 的 62.20%。主要原因是：

- 第一跳经常返回一个长候选列表；
- 整个列表被字符串化后放进下一跳和 image query；
- 图像检索没有稳定地对候选逐个检索；
- 视觉 verifier 对相似人物、鸟类、logo 和局部属性不稳定；
- 视觉答案通常就是最终答案，没有后续文本跳纠正它。

这正是候选路径机制最应该解决的场景：每个文本候选建立独立路径、独立检索图像、独立视觉
验证，再比较，而不是把十几个候选拼成一个超长 query。

## 9. 最终判断

`cor_simple` 不够好不是单一原因，错误规模从大到小主要是：

```text
单答案 unknown 污染
> 已有正确证据但链路/最终选择丢失
> 视觉证据抽取与候选选择错误
> 输出不够精确
> 图像漏召回
> 下游文本阅读错误
> Planner 结构错误
```

Planner 的结构质量已经较高，换更强 Planner 不会自动修复大多数错误。Planner 仍需加强的是
候选集合优先的语义规划，但最主要的工程和方法瓶颈已经转移到：

1. 多候选状态传播；
2. 候选级图像检索；
3. 细粒度视觉验证；
4. 后续证据对前序候选的重排；
5. 终端答案槽位与受限 final selection。

这也是引入 Logic Harness 的直接依据，但 Harness 是否有效必须在固定同一 Planner 输出的条件
下重跑完整 469，不能用 5 题调试集结果代替完整结论。
