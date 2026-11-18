# cor_candidate Logic-Harness 当前设计与实现说明

> 历史说明：该实现最初由 `cor_simple` 的环境开关启用。现在已经拆为独立
> `--method cor_candidate`；`--method cor_simple` 专指与原文本 CoRProRAG
> 对齐的自然语言 LEG + 多模态 SGN + Adaptive Corrector。

本文档说明当前 `cor_candidate` 方法的实际代码设计：面对一个 MMQA 问题时，系统如何分解子问题、如何选择检索策略、如何检索文本和图像、KG 在哪里起作用、候选路径如何保留与重排，以及最后如何生成答案。

当前版本的目标不是复现复杂 typed LEG 版 CoRProRAG，而是测试一个更朴素但带候选状态的方向：

```text
自然语言子问题分解
-> 每个子问题由 LLM 自选检索策略
-> 文本/图像/KG/VLM 证据生成候选
-> 多个候选路径并行保留
-> 后续 hop 对前序候选反向约束和重排
-> 从最佳证据路径生成最终答案
```

当前代码的主链路可以压缩为：

```text
原问题
  |-- gpt-4o-mini：只分解自然语言子问题和依赖
  |-- Answer Contract：抽取最终答案类型与目标关系
  v
逐子问题执行
  |-- Qwen3-VL：动态选择 text/image 组合策略
  |-- BGE + KG：检索文本、表格、实体和关系证据
  |-- CLIP + image-text + KG links：召回外部知识库图像
  |-- Qwen3-VL：看图、验证视觉条件、回答子问题
  v
Candidate -> ReasoningPath 展开、打分和剪枝
  v
按 Answer Contract 选择路径 -> 受限 final selection -> verification -> 短答案
```

## 1. 总体定位

`cor_simple` 和旧 typed CoR 的最大区别是：

- Planner 只输出自然语言子问题和依赖关系。
- Planner 不输出 `operation_type`、`modality`、`edge_type`、`input_source`。
- 执行阶段不再固定使用 `TEXT_RETRIEVE`、`IMAGE_RETRIEVE`、`VISUAL_RELATION` 等显式算子图。
- 每个子问题执行时，再由 LLM 根据当前子问题和已有路径绑定决定检索策略。
- 系统保留多个候选，不把第一跳的单个答案硬塞进后续问题。

也就是说，当前设计想验证的是：

> 不强行把问题塞进复杂 operation taxonomy，只让模型分解自然语言子问题，再用候选路径机制约束它，是否比 typed LEG 更稳。

## 2. 运行入口

主要入口在：

- `src/App.py`
- `src/pipeline/query_executor.py`
- `testCode/Evaluation/GenerateAnswers.py`

当运行参数为：

```bash
--method cor_simple
```

时，系统会调用：

```python
cor_simple_query(...)
```

当前 held-out 469 实验使用的 Slurm 脚本是：

```text
scripts/sbatch_run_cor_simple_gpt4omini_planner_smoke.sh
```

该脚本会再调用统一的：

```text
scripts/sbatch_run_mmqa_corprorag_sanity.sh
```

当前统一配置大致为：

- 数据集：`Input/datasets/multimodalqa/corprorag_v2_strict/mmqa_strict_crossmodal_heldout469.json`
- KG/索引：`Output/indexes/multimodalqa_mmqa_strict_crossmodal_dev569_full_kg`
- 方法：`cor_simple`
- Reader/solver/final：本地 Qwen3-VL OpenAI-compatible 服务
- Planner：APIyi 的 `gpt-4o-mini`
- Text embedding：BGE-M3
- Image embedding：CLIP
- Reranker：BGE reranker

## 3. 模型和服务分工

当前版本不是所有步骤都用同一个模型。

### 3.1 Planner

Planner 使用 `gpt-4o-mini`，只负责分解问题：

```json
{
  "reasoning_plan": "...",
  "sub_questions": [
    {
      "id": 1,
      "question": "...",
      "depends_on": []
    }
  ]
}
```

Planner 不应该输出任何 operation 类型。它的任务只是判断：

- 这个问题是否需要拆成多跳；
- 每一跳自然语言上应该问什么；
- 后续子问题依赖哪些前序答案。

### 3.2 Qwen3-VL

Qwen3-VL 是当前主要本地 reader：

- 子问题检索策略选择；
- 文本证据上的子问题回答；
- 图像视觉观察；
- 图像视觉属性验证；
- 最终答案选择和格式化。

当前脚本里设置：

```bash
USE_VLM_AS_LLM=1
```

因此本来指向文本 LLM 的调用也会被路由到 Qwen3-VL。

### 3.3 BGE-M3

BGE-M3 用于文本向量检索：

- 原问题或子问题检索文本 chunk；
- 图像 caption/title/nearby text 的文本侧检索；
- KG 文本节点或实体描述相关检索。

### 3.4 CLIP

CLIP 用于真正的图像向量检索：

- 用 query 文本编码到 CLIP 共享空间；
- 检索图像 embedding；
- 返回 top image candidates；
- 这些候选之后会被 Qwen3-VL 直接看图验证。

也就是说，当前图像检索不是只用 caption 代替图片。caption/title/nearby text 仍可作为补充文本证据，但图像候选本身应走 CLIP 图像库。

## 4. 面对一个问题的完整处理流程

下面按代码实际顺序说明。

### 4.1 输入问题

输入包含：

- `question_id`
- `question`
- gold answers，仅用于评测
- 可选 query image path
- MMQA corpus/KG/vector index

对于 held-out 469，多数问题没有 query image，而是需要从外部知识库中检索相关图像。

### 4.2 简化子问题分解

系统调用 `simple_question_decomposition` prompt。

输出只有：

- `reasoning_plan`
- `sub_questions`
- `depends_on`

例如，一个问题可能被拆成：

```json
{
  "sub_questions": [
    {
      "id": 1,
      "question": "Which films or TV works is Vandit Bhatt associated with?",
      "depends_on": []
    },
    {
      "id": 2,
      "question": "Which of these works has a poster matching the visual clue?",
      "depends_on": [1]
    }
  ]
}
```

注意，这里不会出现：

- `IMAGE_RETRIEVE`
- `TEXT_RETRIEVE`
- `VISUAL_IDENTIFY`
- `VISUAL_RELATION`
- `CROSS_MODAL_LINK`

这些 typed operation 是复杂 CoR 的设计，不属于 `cor_simple` planner 输出。

### 4.3 Answer Contract

系统会调用 `simple_answer_contract_prompt`，从原问题抽取一个轻量答案契约：

```json
{
  "answer_type": "...",
  "target_relation": "...",
  "answer_slot": "final_answer",
  "cardinality": "one",
  "rationale": "..."
}
```

它的作用是给最终答案提供约束，例如：

- 最终答案应该是人名、学校名、电影名、数字还是 yes/no；
- 答案数量是一个还是多个；
- 问题真正问的是哪个关系。

Answer Contract 不是硬规则推理器，目前主要用于 final 阶段防止模型把中间实体误当最终答案。

### 4.4 初始化 Logic Harness

如果运行：

```bash
python testCode/Evaluation/GenerateAnswers.py --method cor_candidate ...
```

系统会启用候选路径机制。

核心数据结构在：

```text
src/pipeline/simple_logic.py
```

包括：

- `Candidate`
- `SubQuestionState`
- `ReasoningPath`
- `PathManager`
- `AnswerContract`

初始化时只有一条空路径：

```text
p0: bindings = {}, score = 0
```

后续每个子问题不会只产生一个答案，而是产生若干候选，并扩展成多条路径。

### 4.5 一个问题从输入到答案的具体例子

以这类问题为例：

```text
What role did Vandit Bhatt play in the television show that has a poster
featuring a white building with a domed roof?
```

系统希望完成的不是把整句话一次性编码后直接搜图，而是下面的过程。

第一步，Planner 只做自然语言分解：

```json
{
  "sub_questions": [
    {
      "id": 1,
      "question": "What television shows did Vandit Bhatt appear in?",
      "depends_on": []
    },
    {
      "id": 2,
      "question": "Which of {answer_1} has a poster featuring a white building with a domed roof?",
      "depends_on": [1]
    },
    {
      "id": 3,
      "question": "What role did Vandit Bhatt play in {answer_2}?",
      "depends_on": [2]
    }
  ]
}
```

第二步，子问题 1 应选择文本检索。BGE/KG 返回 Vandit Bhatt 关联的电视作品，
执行器从回答、文本证据和实体中生成多个候选，而不是立刻只保留一个：

```text
p1: answer_1 = Tech
p2: answer_1 = Madam Secretary
p3: answer_1 = Quantico
```

第三步，子问题 2 在不同路径上分别展开。对于每个作品名，系统构造作品相关的短图像
查询，通过 CLIP 图像库和 image-text 通道召回 poster，再利用 page title、linked entities
和 KG cross-modal links 检查“图片是否确实属于当前候选”。Qwen3-VL 随后只验证视觉条件：

```text
poster 是否清楚展示 white building with a domed roof？
```

视觉验证为 `yes` 的候选获得支持，为 `no` 的候选受到强惩罚，`uncertain` 不应被当成
肯定证据。后续证据因此可以反过来改变子问题 1 的候选排序。

第四步，子问题 3 用选中的作品作为文本检索条件，查询 Vandit Bhatt 在该作品中的角色。
此时作品名是中间实体，角色名才是终端答案候选。

第五步，系统从保留下来的全路径中选择与 Answer Contract（本题为角色/文本值）一致的
最佳路径，将该路径中的终端候选交给受限 final selector。Final LLM 只能从允许值中选择，
不能凭空生成另一个作品或角色。

这个例子体现三个关键原则：

- 文本约束不应污染 CLIP 图像查询，先用文本得到候选，再逐个找图。
- 图像不是孤立文件；page title、linked entity 和 KG link 用于确认图像属于哪个候选。
- 第一跳候选不是最终答案，后续视觉和文本证据应能对前序候选重新排序。

## 5. 子问题执行流程

每个子问题会在一个或多个 active paths 上执行。

如果子问题含有 `{answer_1}` 这类占位符，系统会用当前路径中的绑定替换它。

关键点是：

- 如果前序答案是 unknown，不会把 `"I don't know"` 直接替换进后续子问题。
- 如果前序子问题有多个候选，后续子问题会分别在不同候选路径上执行。

这就是当前版本相对旧 `cor_simple` 最重要的变化。

### 5.1 检索策略选择

每个 resolved sub-question 会先调用 `simple_retrieval_decision_prompt`。

模型只能选择以下策略之一：

- `text_only`
- `image_only`
- `text_then_image`
- `image_then_text`
- `both_parallel`
- `direct_answer_if_enough`

策略含义如下。

`text_only`：

只检索文本、表格、实体描述等 textual evidence。

`image_only`：

只检索图像，并让 Qwen3-VL 看图。

`text_then_image`：

先通过文本检索得到候选实体或候选集合，再用这些候选去检索图像。

`image_then_text`：

先检索图像，得到图像对应的 page title、linked entities、caption，再用这些实体或标题去查文本。

`both_parallel`：

文本和图像并行检索，再把证据一起交给子问题回答器。

`direct_answer_if_enough`：

只有当已有路径绑定或前序答案足够回答当前子问题时才用。否则容易产生幻觉，所以 trace 里会记录证据不足。

## 6. 文本检索怎么做

文本检索入口大致是：

```python
_simple_text_retrieve(...)
```

内部复用现有 CoRProRAG 的文本检索能力，而不是重新写一个纯 BM25 检索器。

它会调用已有的文本 SGN/RAG 逻辑，使用：

- BGE-M3 embedding；
- reranker；
- KG/text fusion；
- beam context；
- corpus 中的 text/table/image textual side 信息。

返回内容包括：

- 文本答案；
- 文本证据；
- 检索到的 chunk id/doc id；
- 置信度；
- 原始 retrieval trace。

文本和表格目前都属于 textual modality。也就是说，表格列级结构没有被单独建成一个强 typed table operator，这是当前限制之一。

## 7. 图像检索怎么做

图像检索入口大致是：

```python
_simple_image_retrieve(...)
```

内部复用当前 CoRProRAG 的 image retrieval。

一个图像候选通常包含：

```json
{
  "image_id": "...",
  "image_path": "...",
  "page_title": "...",
  "linked_entities": ["..."],
  "caption": "...",
  "global_description": "...",
  "visual_score": 0.0,
  "text_score": 0.0,
  "fusion_score": 0.0
}
```

当前图像检索实际由多路证据融合：

1. CLIP 图像向量检索。
2. 图像 title/caption/nearby text 的文本侧检索。
3. 图像节点到 KG entity 的链接。
4. 图像节点到相关文本节点的 cross-modal links。

其中 CLIP 是图像检索必须存在的部分。caption/title 不是替代图像，而是辅助召回和解释。

## 8. KG 在哪里起作用

当前 `cor_simple` 中，KG 的作用是 evidence enrichment 和 candidate metadata，不是硬编码路径推理器。

### 8.1 文本侧 KG 增强

文本检索不只是从 corpus 里拿 top chunks，还会利用已有 KG：

- entity；
- relation；
- graph neighborhood；
- KG-linked context；
- beam context。

这些信息进入文本证据，让子问题回答器看到更完整的背景。

### 8.2 图像侧 KG 增强

每个图像候选会尽量附带：

- page title；
- linked entities；
- visual entities；
- caption；
- global description；
- graph paths；
- cross-modal links；
- 对应文本 chunk。

这使得一个图像不只是一个文件路径，而是一个可和文本实体、页面、关系相连的证据节点。

例如图像检索得到一个 logo 图像后，系统可以知道：

- 这张图像属于哪个 page；
- 可能链接到哪个学校/球队/电影；
- 与哪些文本 chunk 或 KG entity 相邻。

这就是 KG 对图像检索的增强作用。

### 8.3 当前没有做的事情

当前 `cor_simple` 还没有把 KG 当成严格的 symbolic path verifier。

也就是说，它不会强制执行：

```text
image_node -> visual_entity -> text_entity -> relation -> answer
```

这种硬路径约束。当前是把这些信息整理成证据和候选元数据，让 LLM 和 PathManager 使用。

## 9. 视觉观察和视觉验证

只要策略涉及图像，系统会进一步调用 Qwen3-VL。

主要有三类视觉步骤。

### 9.1 普通视觉观察

入口大致是：

```python
_simple_visual_observe(...)
```

它会把 top image candidates 直接交给 Qwen3-VL，让模型描述与当前子问题相关的视觉信息。

输出进入：

```text
visual_observations
```

### 9.2 视觉属性验证

入口大致是：

```python
_simple_visual_verify(...)
```

当子问题中出现明显视觉谓词，例如：

- logo contains an open book
- bird has stripes above and below the eye
- poster shows a person
- emblem has a lion

系统会把图像和视觉谓词交给 Qwen3-VL 判断 yes/no。

这一步解决的是：

> 图像检索召回了一批候选，但到底哪张满足视觉条件？

### 9.3 候选集合视觉搜索

入口大致是：

```python
_simple_candidate_set_visual_search(...)
```

当问题形如：

```text
Which of these candidates has visual property X?
```

系统会做更细的处理：

1. 从文本或问题中识别候选集合。
2. 为每个候选单独构造图像检索查询。
3. 检索该候选相关图像。
4. 只接受 page title 或 linked entities 与候选名匹配的图像。
5. 对匹配图像做视觉属性验证。

这个 match guard 是为了避免：

```text
用候选 A 的名字检索，却拿到了候选 B 的图像，然后把 B 的视觉属性错记到 A 身上。
```

## 10. 候选如何生成

每个子问题执行后，系统会从多个来源抽取候选：

- 子问题最终 answer；
- text retrieval answer；
- image retrieval answer；
- image candidate 的 page title；
- image candidate 的 linked entities；
- visual verification 中被判定为 yes 的候选；
- selector/disambiguation 选出的候选。

候选结构是：

```json
{
  "value": "...",
  "score": 0.0,
  "confidence": 0.0,
  "evidence_ids": ["..."],
  "source_modality": ["text", "image", "vision"],
  "metadata": {
    "image_id": "...",
    "image_path": "...",
    "page_title": "...",
    "linked_entities": ["..."],
    "visual_score": 0.0,
    "text_score": 0.0,
    "fusion_score": 0.0
  }
}
```

候选不是最终答案。它只是当前路径中的一个可能绑定。

## 11. PathManager 如何工作

`PathManager` 负责维护多条候选路径。

默认配置：

```bash
COR_SIMPLE_CANDIDATE_TOP_K=3
COR_SIMPLE_MAX_ACTIVE_PATHS=5
COR_SIMPLE_CONTRADICTION_PENALTY=0.35
COR_SIMPLE_BACKWARD_RERANKING=1
```

### 11.1 路径绑定

如果第 1 个子问题产生候选：

```text
The Polar Express
Ali
Madam Secretary
```

系统不会只选 top-1，而是扩展成多条路径：

```text
p1: answer_1 = The Polar Express
p2: answer_1 = Ali
p3: answer_1 = Madam Secretary
```

后续子问题中的 `{answer_1}` 会分别替换成这些候选。

### 11.2 路径打分

路径分数由多项启发式组合。当前代码中的核心形式为：

```text
raw = 0.25 * retrieval_score
    + 0.22 * answer_confidence
    + evidence_support
    + 0.10 * visual_support
    + 0.25 * visual_verification
    + 0.10 * entity_consistency
    - contradiction_penalty * contradiction
```

`evidence_support` 当前由离散奖励组成：候选带 evidence id 加 `0.15`，当前 hop 有文本证据
加 `0.12`，有图像证据再加 `0.12`。之后还会应用答案类型乘数，明显不符合答案类型的候选
会被降权。启用 backward reranking 时：

```text
new_path_score = 0.60 * raw + 0.40 * previous_path_score
```

因此当前“反向重排”的准确含义是让后续 hop 的证据改变整条路径排序，还不是显式的符号
约束求解。

参与打分的信息包括：

- retrieval score；
- answer confidence；
- 是否有文本证据；
- 是否有图像证据；
- 是否有视觉观察支持；
- 是否通过视觉验证；
- 与前序绑定是否一致；
- 是否有 contradiction penalty。

目前这不是学习出来的 scorer，也没有在 held-out 上做过系统校准。

### 11.3 反向重排

如果后续子问题发现某条前序候选路径证据更强，它可以超过原来第一跳排名更高的路径。

这就是当前 “Logic Harness” 的核心价值：

> 第一跳检索到多个候选时，不急着确定；后面的 textual/visual evidence 可以反过来重排第一跳候选。

## 12. 最终答案如何生成

最终答案不再完全让 LLM 自由生成。

流程是：

1. `best_path_for_contract()` 在 active paths 中优先选择包含答案类型匹配 binding 的路径。
2. `final_answer_from_path()` 依次尝试显式 answer slot、最新 contract-matching binding、最新
   非空 binding 和最后一个可用子问题答案。
3. `_simple_select_final_answer()` 汇总 selected path 与各子问题的允许值，让 Qwen3-VL 只能
   在这些值中选择，不能自由生成新答案。
4. `verify_final_answer()` 检查答案非空、答案出现在 selected path binding 中、路径状态可用。
5. `_simple_compact_final_answer()` 将冗长回答压缩成评测需要的短答案。

这一步的设计是为了减少：

- final LLM 覆盖掉正确路径；
- 把中间实体误当最终答案；
- 输出一长串解释而不是短答案；
- 输出 `I don't know` 但路径里其实有可用答案。

当前这里仍有一个重要缺口：verification 只验证“答案来自所选路径”，还没有严格验证它
来自负责回答原问题的终端子问题。S. Cole/Muriel 样例中，路径里同时有作品标题和动作描述，
系统仍可能把中间标题或错误视觉描述当作最终答案并通过验证。后续需要给 candidate 增加
`intermediate_entity / retrieval_seed / terminal_answer / support_only` 等明确角色，并把
Answer Contract 绑定到具体终端子问题。

## 13. 输出 trace 里应该看什么

每题输出会包含：

- `simple_plan`
- `answer_contract`
- `subquestion_results`
- `execution_trace`
- `logic_harness`
- final answer/rationale

重点看：

### 13.1 simple_plan

检查 Planner 是否只是自然语言子问题。

如果这里出现 operation type，说明跑错方法或 prompt 被污染。

### 13.2 subquestion_results

每个子问题应有：

- resolved question；
- chosen strategy；
- text evidence；
- image evidence；
- visual observations；
- candidate_state；
- selected candidate；
- failure type。

### 13.3 logic_harness.paths

这里可以看到：

- active paths；
- 每条路径绑定了哪些 `answer_i`；
- 每条路径分数；
- 每条路径证据 id；
- selected path。

这是诊断 “为什么最后选错” 的关键位置。

## 14. 当前系统相对 Flat MM-RAG 的理论优势

Flat MM-RAG 是：

```text
原问题 -> 一次性检索 top text/images -> 全部塞给 VLM -> 直接回答
```

它简单，所以有时很强。尤其当 top-k 直接包含完整证据时，Flat 的优势很明显。

`cor_simple` 想解决 Flat 不擅长的情况：

- 问题需要先找候选集合，再查候选图像；
- 第一跳图像召回多个可能实体，需要后续文本约束重排；
- 第一跳文本候选多个，需要视觉属性筛掉；
- 问题需要 image -> textual 或 textual -> image 的方向切换；
- 单次 flat context 太杂，VLM 容易被无关证据干扰。

所以 `cor_simple` 的价值不在于一次性召回更多，而在于：

```text
把证据组织成候选路径，并让后续 hop 约束前序 hop。
```

## 15. 当前已知失败原因

当前系统仍有几个明确弱点。

### 15.1 视觉 verifier 不够可靠

典型例子是鸟类视觉属性题。

正确图像可能已经进入候选，但 Qwen3-VL 对细粒度视觉特征判断不稳定，例如：

```text
stripe below and above the eye
```

模型可能把多个相似鸟类都判成 yes，导致 selector 选错。

这不是纯检索缺失，而是视觉属性验证能力不足。

### 15.2 子问题分解仍可能不理想

Planner 虽然不输出 operation，但仍可能把问题拆得太粗或太细。

例如它可能把：

```text
先找符合文本条件的学校，再看 logo 是否有 open book
```

拆成一个混合子问题，导致检索 query 太长，图像检索被文本条件污染。

### 15.3 策略选择仍依赖 LLM 判断

每个子问题的检索策略由 Qwen3-VL 判断。

如果模型把视觉条件当成文本条件，可能选择 `text_only`。

如果模型把文本候选集合当成视觉检索 query，可能选择 `image_only` 并导致召回偏移。

### 15.4 图像召回仍有上限

如果正确图像没有进入 CLIP/image_text/KG 融合检索池，后续 PathManager 无法凭空恢复。

Logic Harness 只能重排已有候选，不能创造未召回证据。

### 15.5 KG 目前是弱约束

KG 现在主要是 enrichment，不是强 symbolic verification。

因此当多个候选都有相似文本和图像证据时，KG 不一定能强制筛掉错误路径。

### 15.6 Scorer 仍是启发式

Path score 由规则组合得到，没有经过学习或校准。

如果视觉 verifier 给错高置信度，路径分数也会被带偏。

## 16. 当前版本适合怎么分析结果

看一题失败时，建议按下面顺序查：

1. `simple_plan` 是否把问题拆清楚。
2. 每个 `chosen_strategy` 是否合理。
3. 文本证据里是否有 gold textual evidence。
4. 图像候选 top-k 里是否有 gold image。
5. gold image 如果出现，是否被 visual verifier 判错。
6. gold candidate 是否进入 `candidate_state`。
7. gold candidate 是否进入 active path。
8. 如果进入 active path，为什么 selected path 没选它。
9. 如果 selected path 正确，final answer 是否又被 LLM 改错。

这样可以把错误分成：

- decomposition error；
- strategy error；
- retrieval miss；
- visual verification error；
- candidate extraction error；
- path reranking error；
- final overwrite error。

## 17. 与复杂 typed CoR 的关系

复杂 typed CoR 的问题在于：

- Planner 容易输出错误 operation；
- operation 之间传递的是单步答案；
- 一旦第一跳选错，后续 hop 被污染；
- 一些视觉/文本混合问题被过早归类。

`cor_simple` 现在的改法不是否定 CoR，而是把 CoR 的核心改成：

```text
自然语言子问题 + 动态检索策略 + 候选路径状态
```

而不是：

```text
自然语言子问题 + 固定 operation DAG + 单答案传递
```

后续如果要做论文方法，真正可以强调的方向是：

```text
Logic-Harnessed Multimodal Multi-hop RAG
```

即用轻量逻辑状态管理，把多模态检索和 VLM 判断组织成可审计、可重排、可反向约束的候选路径。

## 18. 可提升和可消融的参数

参数应分成三层调节。先调 Logic Harness 的搜索宽度，再调底层检索预算，最后才调整模型
上下文。不要一次同时改变多层，否则无法判断提升来自方法还是更大的检索预算。

### 18.1 Logic Harness 已接入且实际生效的参数

| 环境变量 | 默认值 | 当前作用 | 增大或开启的收益 | 主要代价/风险 |
|---|---:|---|---|---|
| `COR_CANDIDATE_TOP_K` | `3` | 每个 base path 最多扩展多少个可绑定候选 | 提高 gold candidate survival | 路径数、检索调用和噪声增加 |
| `COR_CANDIDATE_MAX_ACTIVE_PATHS` | `5` | 每一 hop 后最多保留多少条路径 | 减少前序正确路径被剪掉 | 成本近似随路径数增长 |
| `COR_CANDIDATE_MIN_SCORE` | `0.0` | 候选生成阶段的最低分数阈值 | 提高阈值可去掉弱候选 | 未校准前容易误删 gold |
| `COR_CANDIDATE_CONTRADICTION_PENALTY` | `0.35` | 对 `CONTRADICTED` 路径施加惩罚 | 更快淘汰明确冲突路径 | verifier 误判时会放大错误 |
| `COR_CANDIDATE_BACKWARD_RERANKING` | `1` | 将前序路径分数混入后续 hop 排序 | 保留跨 hop 一致性 | 前序高分错误可能形成惯性 |
| `COR_SIMPLE_VISUAL_VERIFIER` | `1` | 对视觉谓词执行 Qwen3-VL yes/no/uncertain 验证 | textual→image 题需要 | 增加 VLM 调用且有细粒度误判 |
| `COR_CANDIDATE_STRUCTURED_FINAL` | `1` | 启用受限 final selection | 降低 final 自由改写 | 上游 allowed values 错时无法自救 |

建议第一轮搜索以下组合：

```text
candidate_top_k: 3 / 5
max_active_paths: 5 / 8 / 10
contradiction_penalty: 0.15 / 0.35 / 0.50
```

优先观察 Gold Path Survival Rate、平均每题 VLM 调用次数和延迟，而不是只看 5 题 EM。
如果 gold 常在候选第 4、5 位，将 `candidate_top_k` 从 3 提到 5 更合理；如果 gold 已进入
路径但被后续剪掉，则优先提高 `max_active_paths` 或调整 scorer，而不是继续扩大图像召回。

### 18.2 底层文本、KG 和图像检索参数

当前 MMQA Slurm 通用脚本覆盖的主要默认值为：

| 环境变量 | 脚本默认值 | 作用 | 调优建议 |
|---|---:|---|---|
| `KG_TOP_K` | `20` | 文本 chunk 检索池 | 与 baseline 公平对比时固定 |
| `KG_ETOP_K` | `20` | KG entity 检索池 | 实体漏召回时尝试 30/40 |
| `KG_RTOP_K` | `20` | KG relationship 检索池 | 关系型问题漏路径时提高 |
| `IMG_TOP_K` | `20` | CLIP 图像向量初始召回池 | gold image 不在池中时提高到 40 |
| `IMAGE_TEXT_TOP_K` | `20` | title/caption/nearby-text 图像侧召回池 | 标题明显但 CLIP 弱时提高 |
| `IMAGE_RETRIEVAL_CANDIDATE_K` | `20` | 图像融合/重排候选池 | 应大于最终 image evidence 数量 |
| `IMAGE_RETRIEVAL_VISUAL_WEIGHT` | `0.45` | CLIP 视觉分数在图像融合中的权重 | 纯视觉属性题可试 0.55/0.65 |
| `MM_CANDIDATE_STATE_TOP_K` | `5` | typed 多模态执行器保留候选数 | 检查复用函数是否影响当前路径 |
| `MM_SGN_CONTEXT_MAX_TOKENS` | `5000` | 单次 SGN/检索回答上下文预算 | 超长报错时降低，证据截断时提高 |
| `IMAGE_CONTEXT_MAX_CHARS` | `3500` | 单图文本/KG 描述最大字符数 | 防止图像 metadata 挤占上下文 |

公平比较 Flat、QD、MMGraph、MC 与 CoR 时，以上检索预算应该固定。方法内部路径数量可以作为
方法特有参数，但不能通过给 CoR 更大的 CLIP/text top-k 来制造不公平提升。

### 18.3 模型和执行成本参数

| 参数 | 当前常用值 | 作用与建议 |
|---|---:|---|
| `LLM_MAX_TOKENS` | `2048` | 子问题与 final 输出上限；当前任务通常不需更大 |
| `VLM_MAX_TOKENS` | 脚本为 `1024` | 视觉观察输出上限；应控制简短，避免吞吐下降 |
| `CONCURRENCY` | 通常 `1` | 并发题数；服务稳定后可逐步提高，但会放大超时与显存压力 |
| Planner model | `gpt-4o-mini` | 只影响自然语言分解，不应影响 reader/backbone 公平性 |
| Reader/final model | Qwen3-VL-32B | baseline 与 CoR 应统一，避免把 backbone 差异当方法提升 |

### 18.4 当前只声明、尚未真正接入执行逻辑的参数

下面两个配置目前在 `src/config.py` 中存在，但执行代码没有读取它们。现在调整不会改变结果：

| 环境变量 | 默认值 | 当前状态 |
|---|---:|---|
| `COR_SIMPLE_MAX_RECOVERY_ATTEMPTS` | `2` | 预留；dependency recovery 当前不是按该次数循环 |
| `COR_SIMPLE_AMBIGUITY_MARGIN` | `0.10` | 预留；`status_from_candidates()` 仍硬编码使用 `0.10` |

后续若要把它们纳入消融，必须先接入 `_simple_recover_dependency_paths()` 和
`status_from_candidates()`，并在 trace 中记录实际生效值。

## 19. 推荐的提升顺序

当前最值得做的不是直接把所有 top-k 调大，而是按错误所在层逐步提升：

1. **修答案角色和终端槽位。** 区分中间实体与终端答案，并让 verification 检查来源 hop。
2. **校准候选与路径 scorer。** 在 20 至 50 题诊断集上统计 gold candidate 的生成、存活、
   被选中三个阶段，分别定位 extraction、pruning 和 final selection 错误。
3. **提升视觉验证。** 对细粒度视觉关系使用候选间 pairwise comparison、目标区域 crop 或
   多次一致性验证，尤其是鸟类、logo 局部元素和人物动作。
4. **再扩大搜索宽度。** 只有 gold 未进入候选或过早被剪掉时，再提高 image pool、
   `candidate_top_k` 或 `max_active_paths`。
5. **最后调上下文和并发。** 这主要影响稳定性、速度和成本，不应被误认为方法创新。

建议每次实验同时报告：EM/F1、gold image recall@K、Gold Path Survival Rate、Final Overwrite
Error Rate、空答案率、平均路径数、平均 VLM 调用次数和单题耗时。这样才能判断某个参数是真的
改善多跳推理，还是仅靠更多检索与更多模型调用换来的结果。
