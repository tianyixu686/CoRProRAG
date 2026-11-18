# CoRProRAG 当前设计与代码执行机制说明

本文档记录的是 **2026-07-29 当前代码实际怎么跑**，不是理想论文版设计。重点是把 CoRProRAG 现在的 planner、LEG、operation、候选状态、最终答案合成、以及已经暴露的问题讲清楚，方便后续从机制上改，而不是继续按单个样例打补丁。

## 1. 当前整体目标

CoRProRAG 当前想做的是：

1. 用 LLM 把原问题分解成一个多模态逻辑执行图，代码里叫 `Modality-aware Logical Execution Graph`，简称 LEG。
2. 每个 LEG 节点有明确 `operation_type`，例如 `TEXT_RETRIEVE`、`IMAGE_RETRIEVE`、`VISUAL_RELATION`、`COMPARE_AGGREGATE`。
3. 每个节点执行一个单跳或局部任务。
4. 节点之间用 `depends_on` 传递中间答案。
5. 新增的 candidate-state 机制试图让上游图像检索不要只输出一个答案，而是保留 top-k 候选，给下游文本检索、视觉关系验证、比较聚合继续约束和重排。
6. 最终把所有节点答案、候选路径和证据交给 LLM 做最终答案合成。

现在最核心的设计分歧是：

- 早期 CoR 更像“单答案链”：每个节点输出一个 `answer`，下游主要用 `{answer_i}`。
- 现在尝试改成“候选状态渐进约束”：`IMAGE_RETRIEVE` 保留多个图像/实体候选，`TEXT_RETRIEVE` 对候选逐个验证，后续证据可以反过来选择前面的图像候选。

但当前实现还没有彻底完成“候选路径作为主状态”的重构，所以存在一些单答案链和候选状态混用带来的问题。

## 2. 入口和执行链

主要入口是：

- `testCode/Evaluation/GenerateAnswers.py`
- 方法参数 `--method corprorag`
- 进入 `src/pipeline/query_executor.py::corprorag_query()`

Slurm 脚本当前通常使用：

- `scripts/sbatch_run_mmqa_corprorag_sanity.sh`

当前 469 任务使用的关键配置是：

- questions: `Input/datasets/multimodalqa/corprorag_v2_strict/mmqa_strict_crossmodal_heldout469.json`
- KG/index: `Output/indexes/multimodalqa_mmqa_strict_crossmodal_dev569_full_kg`
- LLM/VLM: `http://GPU4:18000/v1`, `qwen3-vl-32b-instruct`
- text embedding/rerank/CLIP image embedding: 在 Slurm 任务内 GPU0 本地启动

整体流程：

```text
Question
  -> question_decomposition prompt
  -> parsed LEG
  -> normalize_modality_aware_leg()
  -> LEG postprocess
      -> route_kb_visual_identify_to_image_retrieve()
      -> bypass_redundant_visual_relation()
  -> topological execution
      -> resolve_node_inputs()
      -> execute_mm_sgn_node()
      -> maybe propose_correction_nodes()
  -> collect node_results
  -> candidate_path_summary + answer_focal_node + evidence_parts
  -> final_answer_prompt
  -> final_answer_guard
  -> predicted_answers.json/csv
```

## 3. Planner：问题分解成 LEG

Prompt 在 `src/prompt.py` 的 `PROMPTS["question_decomposition"]`。

当前 planner 要求每个节点包括：

- `id`
- `natural_language_subquestion`
- `operation_type`
- `modality`
- `input_anchor`
- `input_source`
- `expected_output_type`
- `allowed_edge_types`
- `evidence_requirement`
- `depends_on`
- `confidence`

允许的 `operation_type`：

```text
TEXT_RETRIEVE
VISUAL_IDENTIFY
VISUAL_MATCH
CROSS_MODAL_LINK
IMAGE_RETRIEVE
VISUAL_RELATION
GRAPH_EXPAND
COMPARE_AGGREGATE
ANSWER_GENERATE
```

当前 prompt 已加入几个 MMQA 特化规则：

1. 没有真实 `query_image_path` 时，问题里的 logo/photo/image/poster 等一般指 KB/corpus 里的图像，应该用 `IMAGE_RETRIEVE`，不是 `VISUAL_IDENTIFY/query_image`。
2. `logo featuring X`、`open book logo`、`flying horse logo` 这类短语是视觉内容过滤。
3. 如果问题中有文本候选集合，例如 “public schools in Sports in Philadelphia”，这个集合不能交给 CLIP 图像检索去解决。
4. 文本约束应先由 `TEXT_RETRIEVE` 得到候选集合，然后图像节点只处理视觉短语。

例如 q2 类问题应规划成：

```text
TEXT_RETRIEVE:
  Retrieve public schools listed in Sports in Philadelphia / Collegiate sports table.

IMAGE_RETRIEVE:
  Among {answer_1}, retrieve/verify logo images whose visual content features open book.
```

注意：planner 只是 LLM 输出，仍可能出错。因此后面还有 normalize 和 postprocess。

## 4. LEG normalize 与后处理

代码在 `src/pipeline/mm_leg.py`。

### 4.1 normalize_modality_aware_leg()

作用：

1. 把 planner 的原始 JSON 规范化。
2. 重新编号节点为 `1..N`。
3. 修正 `{answer_i}` placeholder。
4. 根据 `depends_on` 自动构建 `D_L`。
5. 填充 `M_L` 和 `T_L`。

如果 planner 不给 `operation_type`，会用 `_infer_operation_type()` 从 modality/query 里猜，这一步有风险，因为猜测规则比较粗。

### 4.2 route_kb_visual_identify_to_image_retrieve()

位置：`src/pipeline/mm_leg.py`

作用：

如果 planner 输出了：

```text
VISUAL_IDENTIFY, input_source=query_image
```

但实际没有 query image，并且问题文本看起来像 KB 图像过滤，例如 logo/badge/open book/flying horse，就把它改成：

```text
IMAGE_RETRIEVE, input_source=kb_image
```

这是为了修之前 “没有 query image 却第一步 visual identify” 的实现错误。

### 4.3 bypass_redundant_visual_relation()

作用：

如果 planner 生成：

```text
IMAGE_RETRIEVE -> VISUAL_RELATION -> TEXT_RETRIEVE
```

而 `VISUAL_RELATION` 只是 “logo featuring/contains” 这类泛化检查，则删除这个视觉关系节点，把下游直接连到 `IMAGE_RETRIEVE`。

原因是当前 `IMAGE_RETRIEVE` 已经承担“根据视觉短语找图并映射到实体”的任务。如果再插一个泛泛的 `VISUAL_RELATION`，容易多一次 VLM 判断和错误传播。

### 4.4 现在的问题

后处理本质是规则补丁，不是 planner 自身能力。它能修几类明显错误，但也说明当前 planner 与 operation 语义还没有完全稳定。

如果后续要做论文方法，最好把这些改成显式机制：

- Planner 必须输出 `visual_constraint` 和 `textual_constraint` 字段，而不是把所有东西混在自然语言子问题里。
- 后处理只做合法性检查，不应该承担主要语义修复。

## 5. 节点执行：execute_mm_sgn_node()

代码在 `src/pipeline/mm_sgn.py::execute_mm_sgn_node()`。

分发逻辑：

```text
TEXT_RETRIEVE / GRAPH_EXPAND -> _run_text_sgn()
VISUAL_IDENTIFY              -> _run_visual_identify()
VISUAL_MATCH                 -> _run_visual_match()
CROSS_MODAL_LINK             -> _run_cross_modal_link()
IMAGE_RETRIEVE               -> _run_image_retrieve()
VISUAL_RELATION              -> _run_visual_relation()
COMPARE_AGGREGATE            -> _run_compare()
ANSWER_GENERATE              -> 空节点，最后统一合成
```

每个节点返回统一结构：

```json
{
  "node_id": 1,
  "operation_type": "IMAGE_RETRIEVE",
  "modality": "image",
  "query": "...",
  "answer": "...",
  "rationale": "...",
  "evidence": [],
  "path": [],
  "confidence": 0.0,
  "failure_type": "",
  "produced_type": "entity",
  "candidate_state": {}
}
```

重要：`answer` 现在仍然存在，而且很多下游逻辑、最终合成仍会读它。这是当前“单答案链”和“候选状态链”混用的关键风险点。

## 6. IMAGE_RETRIEVE 当前怎么做

代码：`src/pipeline/mm_sgn.py::_run_image_retrieve()`

输入通常是一个子问题，比如：

```text
Retrieve KB/corpus images of current America East Conference member schools whose logo features a flying horse.
```

或者：

```text
Among University of Delaware, Delaware State University, Temple University,
retrieve/verify KB logo images whose visual content features at least one open book.
```

### 6.1 分离视觉查询和文本约束

先调用：

```python
_split_image_query_constraints(query)
```

目标：

- `visual_query`: 给 CLIP 图像向量检索使用。
- `textual_constraint_query`: 给 image_text 检索 / candidate exact 约束使用。

例如 q2：

```text
原始 query:
Among University of Delaware, Delaware State University, Temple University,
retrieve/verify KB logo images whose visual content features at least one open book.

visual_query:
logo featuring at least one open book

textual_constraint_query:
Among University of Delaware, Delaware State University, Temple University
```

这一步是最近为了解决“用整句话搜图”问题加的。

### 6.2 候选实体解析

从 `textual_constraint_query` 里解析候选实体：

```python
_candidate_entities_from_constraint()
```

例如得到：

```text
University of Delaware
Delaware State University
Temple University
```

### 6.3 图像检索来源

当前融合四类来源：

1. `visual_results`
   - 来自 `images_vdb.query(visual_query)`
   - 这是 CLIP 图像向量库。
   - 文本 query 进入 CLIP text tower，再查 image embedding。

2. `text_results`
   - 来自 `image_text_vdb.query(visual_query)`
   - 这是图像 caption / global_description / OCR 等文本表示的 BGE 检索。

3. `constraint_text_results`
   - 来自 `image_text_vdb.query(textual_constraint_query)`
   - 用文本约束在图像文本库中找候选图片。

4. `candidate_exact_results`
   - 最近加的候选约束精确召回。
   - 扫描 `image_chunks_db`，如果候选实体名出现在 caption/description/OCR 中，就把对应图片加入候选。
   - 如果同时命中 visual terms，例如 `open book`，分数更高。

这意味着当前 `IMAGE_RETRIEVE` 不是纯 CLIP，也不是纯 caption，而是：

```text
CLIP image retrieval
+ image text retrieval
+ textual constraint image-text retrieval
+ candidate exact recall
+ KG image context enrichment
+ reranker
+ fusion sort
```

### 6.4 KG 上下文扩展

每个图像候选会调用：

```python
_image_kg_context()
```

它做：

1. 根据 image id 取 `image_chunks_db`。
2. 得到 `image_path`、`caption`、`global_description`。
3. 根据图片路径构造图节点名：

```text
IMAGE:{basename(image_path).upper()}
```

4. 在 KG 中找一跳路径。
5. 如果一跳连到 `VISUAL_ENTITY:*`，再从视觉实体找二跳。
6. 收集 linked text entities。
7. 从 caption 中解析 page title，作为候选实体之一。

所以 CoR 里“图像检索能映射到实体”，不是只靠 CLIP；还靠 MMKG 的 image -> visual entity -> text entity/page title 链接。

### 6.5 融合排序

候选合并后会计算：

```text
visual_score_normalized = minmax(CLIP score)
text_score_normalized   = minmax(image_text / constraint_text / candidate_exact score)
rerank_score            = minmax(BGE reranker over candidate text)
lexical_score           = 0.7 * rerank_score + 0.3 * text_score
base_fusion             = visual_weight * visual_score + (1 - visual_weight) * lexical_score
```

其中：

```text
IMAGE_RETRIEVAL_VISUAL_WEIGHT = 0.45
IMAGE_RETRIEVAL_CANDIDATE_K   = 20
MM_CANDIDATE_STATE_TOP_K      = 5
```

如果候选来自 `candidate_exact_results`，当前额外加：

```text
candidate_constraint_hit = 1.0
fusion_score = 1.0 + base_fusion
```

这样 q2 中 “先文本得到学校集合，再视觉判断 open book logo” 可以优先保留 Delaware/Temple 这类候选，而不是被全局 open-book 图像干扰。

### 6.6 IMAGE_RETRIEVE 输出

`answer` 仍然取 top-1 linked entity：

```python
answer = linked_entities[0] if linked_entities else image_id
```

同时输出：

```json
"candidate_state": {
  "type": "image_entity_candidates",
  "selected_answer": "...",
  "selected_answer_tentative": "...",
  "selected_image_id": "...",
  "selection_policy": "tentative_top_fusion_score; downstream nodes may rerank candidates",
  "visual_query": "...",
  "textual_constraint_query": "...",
  "candidate_entities": [],
  "candidates": [
    {
      "rank": 1,
      "image_id": "...",
      "image_path": "...",
      "caption": "...",
      "page_title": "...",
      "global_description": "...",
      "ocr_text": "...",
      "linked_entities": [],
      "visual_score": 0.0,
      "text_score": 0.0,
      "rerank_score": 0.0,
      "fusion_score": 0.0
    }
  ]
}
```

### 6.7 IMAGE_RETRIEVE 当前风险

1. `answer` 仍是 top-1，这会诱导 final answer 或下游过早相信它。
2. `candidate_exact_results` 是规则式补充，会引入偏置。它解决 q2，但不是优雅的通用机制。
3. `visual_query` 分离目前靠正则，对复杂问题不稳。
4. 候选排序没有真正用 VLM 判断视觉属性，只是 CLIP + caption/text + KG/rerank 混合。
5. `candidate_state` 里记录了 top-k，但 `evidence` 里还有 top-10，两个列表可能让 final prompt 混乱。

## 7. TEXT_RETRIEVE 当前怎么做

代码：

- `_run_text_sgn()`
- `_run_text_sgn_once()`

### 7.1 单次文本检索

`_run_text_sgn_once()` 做：

1. 用 `_node_retrieval_query()` 把 node 信息拼成检索 query。
2. 调用 `fuse_kg_text()`：
   - text chunks
   - entities
   - relationships
   - 也可返回 ranked images
3. 对 KG/text 结果做 `beam_search()`。
4. 用 `process_beam_output()` 打包上下文。
5. 调用 `_exact_text_context()` 做 exact table/page fallback。
6. 控制 context tokens 到 `MM_SGN_CONTEXT_MAX_TOKENS`。
7. 调用 `PROMPTS["answer_sub_query"]` 让 LLM 回答子问题。

输出的 `evidence` 是 context 片段，不是精确 span。

### 7.2 候选感知文本验证

`_run_text_sgn()` 会先生成 candidate-aware nodes：

```python
_candidate_aware_text_nodes()
```

如果依赖节点有 `image_entity_candidates`，它会为每个候选实体构造一个文本检索 query。

例如 q5：

```text
原始 node2:
Retrieve the enrollment of Binghamton University, verifying all plausible school candidates from the image step if needed.

候选尝试：
Retrieve the enrollment of SMU Mustangs football ...
Retrieve the enrollment of Al-Zawra ...
Retrieve the enrollment of Binghamton University ...
...
```

每个 attempt 记录：

```json
{
  "query": "...",
  "answer": "...",
  "confidence": 0.9,
  "upstream_fusion_score": 0.78,
  "selection_score": 1.20,
  "evidence_hit": true,
  "negative_verification": false,
  "failure_type": "",
  "candidate_source": {},
  "candidate_entity": "...",
  "candidate_image_id": "..."
}
```

排序公式：

```text
selection_score =
  text_confidence
  + 0.2 * upstream_fusion_score
  + 0.15 if evidence_hit
```

最终选分最高的非 unknown。

### 7.3 TEXT_RETRIEVE 当前风险

1. 它仍然把每个候选变成独立 query，让 LLM 自己从 context 回答，代价高。
2. `evidence_hit` 现在只是 “有 evidence 且非 unknown 且不是 negative verification”，不是严格判断证据真的支持。
3. 对候选的 query 替换靠字符串 placeholder / selected answer 替换，复杂依赖可能不稳。
4. 候选验证只用于下游 text hop，不是全局统一 path state。
5. `candidate_text_answers` 的 `selected_answer` 可以是属性值，如 `17,768`，也可以是实体名；最终合成必须理解问题问什么，否则会把实体和属性混淆。

## 8. VISUAL_RELATION 当前怎么做

代码：`_run_visual_relation()`

优先逻辑：

1. 如果依赖节点有 `image_entity_candidates`，取候选图像集。
2. 对 top-5 图像逐张调用 VLM。
3. prompt 是 `visual_grounding_subquery`。
4. 第一个非 unknown 的 VLM answer 就直接返回。
5. 如果都失败，则返回 attempts。

输出：

```json
"candidate_state": {
  "type": "visual_relation_attempts",
  "attempts": [
    {
      "candidate_image_id": "...",
      "image_path": "...",
      "candidate_entity": "...",
      "visual_answer": "...",
      "relation_confidence": 0.7
    }
  ],
  "selected_attempt": {}
}
```

如果没有候选图像，则退回 KG 一跳 `VISUAL_RELATION`/`CONTAINS` 边。

当前问题：

1. VLM 判断不是批量比较，而是 first non-unknown early return。
2. 没有把视觉关系结果反向更新 `IMAGE_RETRIEVE` 的候选排序。
3. 没有严格 negative/positive 解析。
4. 当前很多 logo featuring 问题被 postprocess 删除了 `VISUAL_RELATION`，所以真正使用视觉关系验证的机会有限。

## 9. COMPARE_AGGREGATE 当前怎么做

代码：`_run_compare()`

优先逻辑：

1. 如果依赖节点有 `candidate_text_answers` 或 `visual_relation_attempts`，构造 candidate path table。
2. 直接按：

```text
evidence_hit
selection_score
confidence
```

排序，取第一条 path 的 answer。

3. 如果没有 candidate path table，就退回比较 dependent node 的答案。
4. 如果能从答案中解析两个数字，并问题里有 lower/higher 等关键词，则比较数字，返回对应 label。
5. 否则做简单 yes/no 或拼接答案。

当前问题：

1. candidate path table 里 path 的 `answer` 可能是中间属性值，而最终问题需要对应实体。
2. q4 暴露过这个问题：比较节点内部曾输出 `10,000`，final LLM 原本能合成 `Amanat Baghdad`，但后处理如果太宽就会把最终答案覆盖成数字。
3. 现在 final guard 已收窄，不会覆盖 `COMPARE_AGGREGATE`，但这说明 compare 本身还不够结构化。
4. 理想状态下 compare 的 candidate path 应包含：

```json
{
  "candidate_entity": "Amanat Baghdad",
  "compared_attribute": "capacity",
  "attribute_value": 10000,
  "opponent_entity": "Naft Al-Wasat",
  "opponent_value": 20000,
  "comparison": "lower",
  "winner_entity": "Amanat Baghdad"
}
```

当前还没有这么规范。

## 10. Final Answer 当前怎么做

代码：`src/pipeline/query_executor.py`。

节点执行完后，构造：

1. `sub_queries_content`
   - 每个节点的 summary。

2. `candidate_path_summary`
   - 从各 node 的 `candidate_state` 抽取 compact path。

3. `answer_focal_node`
   - 找 LEG 的 sink node，即没有被其他节点依赖的节点。
   - 优先 `ANSWER_GENERATE`、`COMPARE_AGGREGATE`、`TEXT_RETRIEVE`、`GRAPH_EXPAND`。

4. `evidence_parts`
   - 先放 `answer_focal_node`
   - 再放 `candidate_path_summary`
   - 再放每个节点的 candidate_state 和 evidence。

然后调用：

```text
PROMPTS["final_answer_prompt"]
```

### 10.1 final answer guard

最近加了一个保护：

```python
_should_override_with_focal_answer()
```

它只在以下条件成立时覆盖 final LLM 输出：

1. final LLM 输出的答案不等于 focal node answer。
2. focal node 是 `TEXT_RETRIEVE`。
3. focal node 有 evidence 或 selected candidate attempt。
4. final LLM 输出的答案恰好是上游图像候选实体。

它的目的只修 q5 这种情况：

```text
node1 IMAGE_RETRIEVE answer = Binghamton University
node2 TEXT_RETRIEVE answer = 17,768
final LLM answer = Binghamton University  # 错，把实体当最终答案
guard override -> 17,768
```

它现在不会覆盖 `COMPARE_AGGREGATE`，避免 q4 从 `Amanat Baghdad` 被覆盖成中间 capacity 数字。

### 10.2 final answer 当前风险

1. 最终答案仍由 LLM 合成，不是完全符号化。
2. candidate_path_summary 可能把上游实体、下游属性、比较结果混在一起。
3. `answer_focal_node` 是 sink heuristic，不一定等于真正最终答案节点。
4. guard 是补救，不是根治。

## 11. 当前几个样例暴露的问题

### 11.1 q2 open-book logo

问题：

```text
What public school in the Collegiate sports of Sports in Philadelphia has a logo that features at least one open book?
```

正确机制应该是：

```text
TEXT: 找 Sports in Philadelphia / Collegiate sports / public schools
IMAGE: 只用 logo featuring open book 判断候选学校 logo
```

之前错误：

```text
把整句 “public school in Collegiate sports ... logo open book” 拿去搜图
```

这让图像检索承担了它无法承担的文本表格约束。

当前修复：

1. planner prompt 加了候选集合 + 视觉短语分离示例。
2. `_split_image_query_constraints()` 把 visual/textual constraint 拆开。
3. `_candidate_exact_image_results()` 让上游候选学校图像进入 image candidate。

当前结果：

```text
node1 answer = University of Delaware, Delaware State University, Temple University
node2 visual_query = logo featuring at least one open book
node2 top candidates = University of Delaware / Delaware State University / Temple University ...
final = University of Delaware
```

仍然的问题：

这是规则和检索融合修出来的，不是一个干净的、可泛化的候选约束模型。

### 11.2 q4 lion logo capacity comparison

问题：

```text
Which team in the 2017-18 Iraqi Premier League has a lower capacity:
the team with a logo featuring a lion or Naft Al-Wasat?
```

当前机制大致是：

```text
IMAGE_RETRIEVE: 找 lion logo team -> Amanat Baghdad
TEXT_RETRIEVE: 找 Amanat Baghdad capacity
TEXT_RETRIEVE: 找 Naft Al-Wasat capacity
COMPARE_AGGREGATE: 比 lower
final: Amanat Baghdad
```

暴露问题：

`COMPARE_AGGREGATE` 内部容易输出中间数值，而不是 winner entity。final LLM 有时能修回来，但如果我们做机械 override 会弄错。

说明：

比较节点需要结构化输出 winner entity，而不是只输出 answer 字符串。

### 11.3 q5 flying horse enrollment

问题：

```text
The current America East Conference member school with flying horse logo had what enrollment?
```

当前机制：

```text
IMAGE_RETRIEVE: 找 flying horse logo candidates
TEXT_RETRIEVE: 对候选学校逐个查 enrollment
final: 17,768
```

暴露问题：

图像 top-1 可能是错的，如 SMU Mustangs football；但 Binghamton 在候选中，下游 TEXT_RETRIEVE 可以查到 `17,768`。如果 final LLM 被上游实体带偏，就会答 `Binghamton University` 而不是 enrollment。

当前 guard 修的是这个最终答案层错误。

## 12. 当前方法的真实优点

1. 可以显式记录每一步：planner、operation、证据、candidate_state。
2. 能看到图像检索、文本检索、候选验证到底哪一步丢了。
3. 对 “先图像后文本” 任务，候选状态可以让后续文本证据把错误 top-1 图像拉回来。
4. 能利用 KG 的 image -> visual entity -> text entity/page title 链接，而不是只把图片当孤立向量。
5. 在 q2 这种“文本候选集合 + 视觉属性”问题中，候选状态有潜力超过 flat RAG，因为 flat RAG 通常只是把证据堆给 VLM。

## 13. 当前方法的真实弱点

### 13.1 Planner 不稳定

即使 prompt 已加强，planner 仍可能：

- 把文本约束和视觉约束混在一个 IMAGE_RETRIEVE query 里。
- 不该分解时强行分解。
- 该分成并行分支时生成串行链。
- 把 query image / KB image 搞混。
- 输出 operation 语义不清的节点。

### 13.2 Operation 输入还不够结构化

现在节点主要还是自然语言字段：

```text
query
input_anchor
expected_output_type
```

缺少显式字段：

```json
{
  "textual_constraints": [],
  "visual_constraints": [],
  "candidate_set": [],
  "target_attribute": "",
  "answer_type": "",
  "selection_variable": ""
}
```

所以 `_split_image_query_constraints()` 只能靠正则，容易不稳。

### 13.3 Candidate state 不是一等公民

虽然现在有：

- `image_entity_candidates`
- `candidate_text_answers`
- `visual_relation_attempts`
- `compare_candidate_paths`

但主执行图仍然围绕每个节点的单个 `answer` 跑。

理想状态应该是：

```text
node output = candidate table / state object
answer = optional display field
downstream always consumes state, not answer string
final answer from selected path, not LLM guess
```

### 13.4 COMPARE_AGGREGATE 结构弱

比较题需要保留实体和值的绑定关系。目前只是在 answer 字符串里解析数字，或者从 candidate path 里取最高分 answer。

这会导致：

- 中间数值和最终实体混淆。
- lower/higher winner 依赖 label 解析。
- 复杂比较、集合交集、计数会不稳。

### 13.5 视觉关系验证不足

现在 logo featuring 这类问题多数直接在 IMAGE_RETRIEVE 阶段处理，`VISUAL_RELATION` 很少真正发挥。

但你的目标是“外部知识库图像视觉关系推理”，那么后续必须让系统能显式做：

```text
candidate images
-> VLM verify visual relation/attribute
-> update candidate scores
-> pass verified candidate state downstream
```

而不是把视觉关系压进 CLIP 检索。

### 13.6 Final answer 仍过度依赖 LLM

final LLM 会把候选实体、属性值、比较中间结果混淆。当前 guard 只是窄修。

理想状态是：

- 若最终 path 已经结构化给出 answer，直接用结构化 answer。
- final LLM 只写 rationale，不决定答案。
- 只有没有结构化答案时才让 LLM 合成。

## 14. 与 flat RAG 的关键区别

Flat RAG：

```text
query -> text/image retrieval top-k -> 全部证据塞给 VLM -> answer
```

优点：

- 少分解错误。
- 少中间答案传播错误。
- 如果 top-k 证据里已经包含所有信息，VLM 可能直接整合得很好。

CoR 当前：

```text
query -> planner -> operation nodes -> candidate propagation -> final
```

潜在优点：

- 可诊断每一跳。
- 可利用后续约束重排前面候选。
- 可对视觉关系做显式验证。
- 可形成“多跳跨模态路径”而不是证据堆。

当前为什么可能不如 flat：

1. 分解错会直接损失。
2. 每个 operation 的 top-1 answer 会早期定型。
3. candidate_state 还没有变成主数据结构。
4. final LLM 会混淆上游实体和最终答案。
5. 视觉关系多数还只是检索短语，没有真正 VLM relation verification。

## 15. 建议的下一轮设计重构方向

### 15.1 Planner 输出结构化约束

不要只输出自然语言 subquestion，改为每个节点输出：

```json
{
  "operation_type": "IMAGE_RETRIEVE",
  "candidate_source": "node_1",
  "textual_constraints": [
    "public school",
    "listed in Sports in Philadelphia Collegiate sports table"
  ],
  "visual_constraints": [
    {"type": "logo_contains", "object": "open book"}
  ],
  "target_variable": "school",
  "expected_output_type": "entity"
}
```

这样就不需要正则猜 “哪部分给 CLIP，哪部分给文本检索”。

### 15.2 Operation 输出统一 candidate table

每个 operation 都输出：

```json
{
  "state_type": "...",
  "rows": [
    {
      "entity": "",
      "image_id": "",
      "text_evidence": [],
      "visual_evidence": [],
      "attributes": {},
      "scores": {},
      "status": "candidate|verified|rejected"
    }
  ],
  "selected_row": 0
}
```

`answer` 只作为展示，不作为下游主输入。

### 15.3 TEXT_RETRIEVE 支持按候选表 join

例如：

```text
input candidates: schools
target attribute: enrollment
output rows: school + enrollment + evidence
```

而不是每个候选 query 输出一个 loose answer。

### 15.4 VISUAL_RELATION 变成 candidate verifier

例如：

```text
input candidates: school logo images
visual predicate: contains open book
output rows: school + visual_pass + rationale + image crop/evidence
```

它应该更新候选表，而不是返回一个独立 answer。

### 15.5 COMPARE_AGGREGATE 输出 winner path

比较节点应该输出：

```json
{
  "winner_entity": "Amanat Baghdad",
  "comparison": "lower_capacity",
  "left": {"entity": "Amanat Baghdad", "capacity": 10000},
  "right": {"entity": "Naft Al-Wasat", "capacity": 20000}
}
```

然后 final 直接取 `winner_entity`。

### 15.6 Final answer 从结构化 path 取答案

最终答案优先级应为：

```text
explicit final structured answer
> compare winner_entity
> selected candidate row target field
> focal text attribute answer
> LLM synthesis fallback
```

不要默认让 LLM 在一堆 summary 里自己决定。

## 16. 建议后续诊断指标

对 469 每题记录：

1. `planner_error`
   - operation 错
   - dependency 错
   - query_image hallucination
   - text/visual constraint mixed

2. `image_retrieval_error`
   - gold image not in visual top-k
   - gold image in top-k but rerank dropped
   - gold entity linked but wrong selected answer

3. `candidate_propagation_error`
   - gold candidate in image candidates but text verification没查
   - text verification查到但 selection没选
   - selected path正确但 final answer错

4. `visual_relation_error`
   - 没调用 VLM relation
   - VLM 判断错
   - 判断对但没有影响候选排序

5. `compare_error`
   - 属性值对但 winner 错
   - winner 对但 final 覆盖错
   - label/entity 绑定丢失

6. `final_synthesis_error`
   - node answer正确但 final错
   - final 使用上游实体而不是最终属性
   - final 使用中间数值而不是 winner entity

## 17. 当前代码文件索引

核心文件：

- `src/pipeline/query_executor.py`
  - `corprorag_query()`: CoR 主流程
  - `_candidate_path_summary()`: 候选路径摘要
  - `_answer_focal_node()`: 最终答案焦点节点
  - `_should_override_with_focal_answer()`: q5 类 final guard

- `src/pipeline/mm_leg.py`
  - `normalize_modality_aware_leg()`
  - `route_kb_visual_identify_to_image_retrieve()`
  - `bypass_redundant_visual_relation()`
  - `resolve_node_inputs()`
  - `topological_nodes()`

- `src/pipeline/mm_sgn.py`
  - `_run_image_retrieve()`
  - `_run_text_sgn()`
  - `_run_visual_relation()`
  - `_run_compare()`
  - `execute_mm_sgn_node()`

- `src/prompt.py`
  - `PROMPTS["question_decomposition"]`
  - `PROMPTS["answer_sub_query"]`
  - `PROMPTS["visual_grounding_subquery"]`
  - `PROMPTS["final_answer_prompt"]`

- `src/config.py`
  - `mm_candidate_state_top_k`
  - `image_retrieval_candidate_k`
  - `image_retrieval_visual_weight`
  - `mm_sgn_context_max_tokens`
  - KG/text/image top-k

## 18. 一句话总结

CoR 当前不是方向错，而是还处在“单答案链 + 候选状态补丁”的混合阶段。真正应该改的是把候选路径变成一等公民：planner 明确拆出文本约束、视觉约束和目标变量；operation 维护候选表；TEXT/VISUAL/COMPARE 都是在同一个候选表上做过滤、补属性、验证和选择；final 不再猜答案，而是从结构化 selected path 取答案。
