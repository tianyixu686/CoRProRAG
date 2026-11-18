# CoRProRAG 当前算子与 KG 增强流程说明

日期：2026-07-30

本文档说明当前 CoRProRAG 在 MMQA 上实际是怎么运行的。重点不是方法概念，而是代码层面的真实流程：

- 当前有哪些 operation / 算子。
- 每个算子输入什么、做什么、输出什么。
- 检索出图像之后发生什么。
- KG 如何增强文本检索和图像检索。
- 当前实现为什么还没有充分体现“渐进式候选状态 + KG/视觉关系路径验证”的优势。

核心代码位置：

- 主流程：`src/pipeline/query_executor.py`
- LEG 规范与后处理：`src/pipeline/mm_leg.py`
- 算子执行：`src/pipeline/mm_sgn.py`
- prompt：`src/prompt.py`
- KG 构建：`src/kg_builder.py`
- 图路径检索：`src/retrieval/beam_search.py`

## 1. 当前整体流程

一次 CoR 查询从 `corprorag_query()` 开始，流程如下：

1. LLM 对原问题做 `question_decomposition`，生成 modality-aware LEG。
2. `normalize_modality_aware_leg()` 标准化 LEG。
3. 执行 LEG 后处理：
   - `route_kb_visual_identify_to_image_retrieve()`
   - `bypass_redundant_visual_relation()`
4. 按依赖关系拓扑执行每个 node。
5. 每个 node 调用一个 operation executor。
6. 每个 node 输出：
   - `answer`
   - `rationale`
   - `evidence`
   - `path`
   - `confidence`
   - `failure_type`
   - 可选 `candidate_state`
7. 所有 node 执行完后，构造：
   - `sub_queries_content`
   - `candidate_path_summary`
   - `answer_focal_node`
   - `context`
8. 调用 `final_answer_prompt`，让 LLM 输出最终 JSON answer。

也就是说，当前 CoR 的核心不是直接 RAG，而是：

> 原问题 -> LEG 节点序列 -> 逐节点检索/视觉/链接/比较 -> final synthesis。

但是当前实现里，很多节点虽然保留了候选状态，最终仍然偏向“每个 node 先选一个局部答案，再向下游传递”。

## 2. LEG 中有哪些 operation

当前 `src/pipeline/mm_leg.py` 定义的 operation 包括：

| Operation | 主要作用 | 默认 modality | 默认边类型 |
|---|---|---|---|
| `TEXT_RETRIEVE` | 检索文本/表格/KG 上下文并回答一个子问题 | text | `TEXT_RELATION` |
| `GRAPH_EXPAND` | 与 `TEXT_RETRIEVE` 类似，偏向图扩展型文本/KG hop | text | `TEXT_RELATION` |
| `VISUAL_IDENTIFY` | 检查用户提供的 query image，识别图中实体/属性 | image | `CONTAINS` |
| `VISUAL_MATCH` | 用 query image 做图搜图，找相似 KB 图像 | image | `VISUAL_MATCH`, `CONTAINS` |
| `IMAGE_RETRIEVE` | 没有 query image 时，从 KB/corpus 图像库检索图像，并映射到实体 | image | `SOURCE`, `DEPICTS` |
| `VISUAL_RELATION` | 对已选/候选图像做 VLM 视觉关系或视觉属性验证 | image | `VISUAL_RELATION`, `CONTAINS` |
| `CROSS_MODAL_LINK` | 把视觉实体或图像锚点链接到文本/KG 实体 | cross_modal | `IDENTITY`, `DEPICTS`, `ADJACENT_TO_TEXT` |
| `COMPARE_AGGREGATE` | 比较、计数、布尔判断、候选路径聚合 | hybrid | `NONE` |
| `ANSWER_GENERATE` | 占位型最终回答节点，目前最终回答由主流程统一处理 | hybrid | `NONE` |

需要注意：

- `VISUAL_IDENTIFY` 的语义应该是“看用户输入的问题图片”。
- 如果问题没有提供 query image，而是问外部 KB 里的 logo、poster、photo，就不应该用 `VISUAL_IDENTIFY/query_image`。
- 这种情况应该用 `IMAGE_RETRIEVE`。

## 3. LEG 后处理做了什么

### 3.1 `route_kb_visual_identify_to_image_retrieve()`

这个函数解决一个老问题：

> Planner 经常把“外部 KB 图像问题”误写成 `VISUAL_IDENTIFY`，仿佛用户提供了 query image。

比如：

```text
Identify the school whose logo has a flying horse.
```

如果没有 `query_image_path`，它会把类似节点从：

```json
{
  "operation_type": "VISUAL_IDENTIFY",
  "input_source": "query_image"
}
```

改成：

```json
{
  "operation_type": "IMAGE_RETRIEVE",
  "input_source": "kb_image"
}
```

触发条件包括：

- 没有 query image；
- query 里出现 `logo`、`badge`、`emblem`、`seal`、`photo`、`image`、`picture`、`figure`、`open book`、`flying horse` 等 KB visual terms。

这一步只是纠正 operation 类型，不负责真的判断视觉属性。

### 3.2 `bypass_redundant_visual_relation()`

这个函数解决另一类 planner 冗余：

```text
IMAGE_RETRIEVE -> VISUAL_RELATION -> TEXT_RETRIEVE
```

如果中间的 `VISUAL_RELATION` 只是泛泛地检查 logo contains/features，且 `IMAGE_RETRIEVE` 已经输出了实体，代码会绕过这个 `VISUAL_RELATION`，让后面的文本节点直接依赖图像检索结果。

初衷是避免这样的错误链：

```text
IMAGE_RETRIEVE 找到 Amanat Baghdad
VISUAL_RELATION 又问“检查这个 logo 是否有 lion”
VLM 反而输出一个视觉描述或错误实体
TEXT_RETRIEVE 拿错误答案继续查
```

但副作用是：

- 很多真实需要视觉验证的节点也被绕过；
- 最新 469 trace 中 `VISUAL_RELATION` 只出现 9 次；
- 大多数视觉谓词实际上仍由 `IMAGE_RETRIEVE` 的 CLIP/rerank 排名承担。

这也是当前 CoR 效果没有明显提升的重要原因。

## 4. `TEXT_RETRIEVE` 做什么

代码位置：

- `_run_text_sgn()`
- `_run_text_sgn_once()`

### 4.1 单次文本检索 `_run_text_sgn_once()`

它会调用：

```python
fuse_kg_text(...)
```

输入是当前子问题 query，输出包括：

- `ranked_chunks`
- `ranked_entities`
- `ranked_relationships`
- 可选 `ranked_images`

然后它会基于这些结果做 KG beam search：

```python
beam_search(...)
```

beam search 会沿着 KG 边扩展路径，边类型权重来自配置：

| 边类型 | 权重 |
|---|---:|
| `IDENTITY` | 1.0 |
| `DEPICTS` | 0.7 |
| `VISUAL_RELATION` | 0.6 |
| `CONTAINS` | 0.5 |
| `ADJACENT_TO_TEXT` | 0.4 |

随后 `process_beam_output()` 把 beam path、实体、关系、文本 chunks 组织成 context，再调用 `answer_sub_query` prompt，让 LLM 回答当前子问题。

所以 `TEXT_RETRIEVE` 不是纯文本向量检索，它里面有 KG 路径增强：

```text
query
 -> text/entity/relation retrieval
 -> KG beam search
 -> KG path + text chunks context
 -> LLM sub-answer
```

### 4.2 候选感知文本检索 `_run_text_sgn()`

如果上游节点有 `candidate_state`，尤其是 `image_entity_candidates`，`TEXT_RETRIEVE` 会尝试对候选逐个构造查询。

例如上游 `IMAGE_RETRIEVE` 得到候选：

```text
The Polar Express
Harlem Nights
Blood and Bone
...
```

下游节点原本是：

```text
Retrieve the role Nona Gaye played in {answer_2}.
```

代码会生成多个 attempt：

```text
Retrieve the role Nona Gaye played in The Polar Express.
Retrieve the role Nona Gaye played in Harlem Nights.
Retrieve the role Nona Gaye played in Blood and Bone.
...
```

每个 attempt 输出：

- `query`
- `answer`
- `confidence`
- `upstream_fusion_score`
- `selection_score`
- `evidence_hit`
- `candidate_entity`
- `candidate_image_id`
- `failure_type`

然后按：

```text
evidence_hit 优先
selection_score 次之
confidence 再次
```

选择一个 `best_result`。

最后写入：

```json
{
  "type": "candidate_text_answers",
  "selected_query": "...",
  "selected_answer": "...",
  "selected_candidate_entity": "...",
  "selected_candidate_image_id": "...",
  "attempts": [...]
}
```

### 4.3 当前问题

这个机制已经比早停强，但仍然不是真正的全局路径搜索：

- 它最多只验证前 `MM_CANDIDATE_STATE_TOP_K` 个候选。
- 如果正确候选没进入上游候选 top-k，下游救不回来。
- 选择函数主要看文本 answer 是否非 unknown，不理解原问题里的全局约束。
- 它不会把视觉谓词验证结果和文本验证结果做联合打分。

例如 thrush 题里：

- 文本节点拿到了候选鸟列表，里面有 gold 相关 `Redwing`。
- 图像节点 top-5 却是 `Dusky thrush`、`Swainson`、`Eyebrowed thrush` 等。
- 下游文本验证只能在这些错误候选里找 binomial，最后得到 `Turdus obscurus`。

这说明 candidate_state 有了，但候选生成/视觉过滤阶段仍然容易丢 gold。

## 5. `IMAGE_RETRIEVE` 做什么

代码位置：

- `_run_image_retrieve()`
- `_split_image_query_constraints()`
- `_candidate_exact_image_results()`
- `_image_kg_context()`
- `_compact_image_candidate()`

这是当前最关键的图像算子。

### 5.1 输入

输入是一个图像相关子问题，例如：

```text
Retrieve the television show whose poster features a white building with a domed roof.
```

或：

```text
Among Delaware State University, University of Delaware, Temple University, retrieve KB images of logos featuring an open book.
```

### 5.2 拆分视觉 query 和文本约束

代码先调用：

```python
_split_image_query_constraints(query)
```

它试图把 query 拆成：

- `visual_query`
- `textual_constraint_query`

例如理想情况下：

```text
原始 query:
Which public school, whose sports compete against Drexel University, has a logo that features at least one open book?

visual_query:
logo featuring open book

textual_constraint_query:
public school whose sports compete against Drexel University
```

但当前拆分仍然比较弱。很多 poster/photo 类问题没有充分拆开，仍然会把文本约束混进 CLIP 图像检索。

例如 S. Cole 题：

```text
Retrieve the KB/corpus image of the TV title cover where S. Cole played Muriel.
```

这里视觉部分应该是：

```text
TV title cover / person action
```

文本部分应该先查：

```text
S. Cole played Muriel -> title
```

但当前 `IMAGE_RETRIEVE` 直接拿整句去搜图，导致 top1 是 `Command & Conquer 3: Kane`。

### 5.3 多路图像召回

`IMAGE_RETRIEVE` 不是只搜 caption，也不是只搜 CLIP。它有几路召回：

1. `images_vdb.query(visual_query)`

   这是图像向量库。query 是文本时，用 CLIP 文本塔映射到图文共享空间，然后搜 CLIP image embedding。

2. `image_text_vdb.query(visual_query)`

   这是图像文本描述向量库。一般基于 caption、global description、OCR、附近文本等文本表示，用 BGE 或文本 embedding 检索。

3. `image_text_vdb.query(textual_constraint_query)`

   如果能拆出文本约束，会额外用文本约束搜 image textual index。

4. `_candidate_exact_image_results(candidate_entities, visual_query, image_chunks_db)`

   如果 query 里出现候选实体集合，会扫描 image chunks 的 caption/description/OCR，找包含候选实体名字的图片。

所以当前图像检索是一个 fusion：

```text
CLIP 图像检索
+ 图像 caption/text 检索
+ 文本约束检索
+ 候选实体精确图像召回
```

### 5.4 合并候选

多路召回结果按 image id 合并，记录来源分数：

- `visual_retrieval_score`
- `text_retrieval_score`
- `constraint_text_retrieval_score`
- `candidate_exact_retrieval_score`

### 5.5 用 KG 增强每个图像候选

这是图像检索和 KG 发生关系的关键位置：

```python
item["kg_context"] = await _image_kg_context(item, stores)
```

`_image_kg_context()` 做这些事：

1. 根据 image id 从 `image_chunks_db` 取图像 metadata：
   - `image_path`
   - `caption`
   - `global_description`
   - `ocr_text`
2. 根据 `image_path` 构造图中的 image node 名称。
3. 在 KG 中从 image node 做一跳扩展。
4. 如果一跳连到文本实体，加入 `linked_entities`。
5. 如果一跳连到 `VISUAL_ENTITY:*` 节点，再从这些 visual node 做第二跳扩展。
6. 第二跳里如果连到文本实体，也加入 `linked_entities`。
7. 从 caption/page title 再补一个实体名。
8. 去重，得到最终 `linked_entities` 和 `graph_paths`。

也就是说，图像检索出来之后，KG 的作用是：

```text
image candidate
 -> image node
 -> visual entity / text entity / page title
 -> linked_entities
 -> graph_paths
```

这一步把“图片文件”变成“可以传给文本检索和最终回答的实体候选”。

例如图像候选是某个 poster：

```text
image_path = .../polar_express.jpg
```

KG 可能补出：

```text
linked_entities = [
  "The Polar Express",
  "The Polar Express (film)",
  "CHRISTMAS TRADITION"
]
```

然后 `IMAGE_RETRIEVE` 的 `answer` 通常取第一个 linked entity。

### 5.6 KG context 还参与 rerank

每个图像候选会被转成文本：

```python
_image_candidate_text(item)
```

里面包含：

- linked entities
- caption
- global description
- OCR
- graph path 中的 src/tgt name
- graph path description
- target description

然后 reranker 会用：

```text
visual_query + textual_constraint_query
```

去重排这些候选文本。

所以 KG 对图像检索的增强有两个层面：

1. **实体映射增强**：图片 -> linked_entities。
2. **重排文本增强**：图片候选文本包含 KG path，reranker 能利用图路径描述。

### 5.7 融合排序

当前融合逻辑大致是：

```text
visual_score = CLIP 图像检索归一化分
text_score = image_text / textual_constraint / candidate_exact 的最高归一化分
rerank_score = reranker 对 image_candidate_text 的归一化分
lexical_score = 0.7 * rerank_score + 0.3 * text_score
fusion_score = visual_weight * visual_score + (1 - visual_weight) * lexical_score
```

其中：

```text
IMAGE_RETRIEVAL_VISUAL_WEIGHT 默认 0.45
IMAGE_RETRIEVAL_CANDIDATE_K 默认 20
MM_CANDIDATE_STATE_TOP_K 默认 5
```

如果候选来自 `candidate_exact_entity`，会加硬约束：

```text
fusion_score = 1.0 + base_fusion
```

目的是优先保留上游文本候选集合里的实体图片。

### 5.8 输出

`IMAGE_RETRIEVE` 输出的 `answer` 是 top image 的 linked entity：

```text
answer = linked_entities[0]
```

同时保存 `candidate_state`：

```json
{
  "type": "image_entity_candidates",
  "selected_answer": "...",
  "selected_answer_tentative": "...",
  "selected_image_id": "...",
  "selection_policy": "tentative_top_fusion_score; downstream nodes may rerank candidates",
  "visual_query": "...",
  "textual_constraint_query": "...",
  "candidate_entities": [...],
  "candidates": [
    {
      "rank": 1,
      "image_id": "...",
      "image_path": "...",
      "page_title": "...",
      "linked_entities": [...],
      "caption": "...",
      "global_description": "...",
      "visual_score": ...,
      "text_score": ...,
      "rerank_score": ...,
      "fusion_score": ...
    }
  ]
}
```

### 5.9 检索出图像以后到底怎样

总结成一句话：

> 当前 CoR 检索出图像后，会用 KG 把图片候选映射成文本实体候选，并把 top-k 图片/实体作为 candidate_state 传给下游；但大多数情况下并不会自动调用 VLM 逐张验证视觉关系。

也就是说：

```text
图像检索结果
 -> KG 补 linked_entities / graph_paths
 -> 生成 candidate_state
 -> 下游 TEXT_RETRIEVE 尝试用候选实体查文本
 -> final answer prompt 看到候选摘要
```

而不是：

```text
图像检索结果
 -> 对每张图用 VLM 检查 open book/lion/stripe/dommed roof
 -> 根据视觉验证结果重排候选
 -> 再查文本属性
```

后者才是我们希望改成的方向。

## 6. `VISUAL_RELATION` 做什么

代码位置：

- `_run_visual_relation()`

它的作用是对已选图片或上游候选图片做 VLM 视觉验证。

### 6.1 输入来源

优先从依赖节点的 `image_entity_candidates` 里取候选图像：

```text
dependency candidate_state.candidates
```

如果没有，就退回：

```text
execution_context["last_image_evidence"]
```

### 6.2 执行方式

它对前 5 张图像逐张调用 VLM：

```python
vlm_model_func(image_path=image_path, prompt=visual_grounding_subquery)
```

每次得到：

- `visual_answer`
- `relation_confidence`
- `rationale`
- `candidate_image_id`
- `candidate_entity`

如果某张图返回非 unknown 答案，就直接返回该 attempt。

输出的 candidate state：

```json
{
  "type": "visual_relation_attempts",
  "attempts": [...],
  "selected_attempt": {...}
}
```

### 6.3 当前问题

这个算子实现了，但实际使用太少。最新 469 trace 中：

```text
VISUAL_RELATION 只出现 9 次
IMAGE_RETRIEVE 出现 532 次
```

这说明绝大多数视觉条件没有被拆成单独的 `VISUAL_RELATION`，而是直接由 `IMAGE_RETRIEVE` 负责。

这会导致：

- CLIP 排名被当成视觉关系判断。
- Reranker 看的是 caption/KG 文本，不是真的看图。
- 图像 top-k 里视觉属性相似但文本约束错误的候选容易胜出。

## 7. `CROSS_MODAL_LINK` 做什么

代码位置：

- `_run_cross_modal_link()`

它的目标是把视觉 anchor 链接到文本/KG 实体。

执行逻辑：

1. 用 `entities_vdb.query(anchor)` 找实体候选。
2. 如果存在严格同名实体，直接返回。
3. 否则从这些实体在 KG 里做一跳扩展。
4. 保留边类型为：
   - `IDENTITY`
   - `DEPICTS`
   - `ADJACENT_TO_TEXT`
5. 选择最高优先级路径。
6. 如果没有路径，就退化为最近实体候选。

典型用途：

```text
visual entity / image title
 -> text entity
```

但现在很多 `IMAGE_RETRIEVE` 已经直接输出 linked entity，所以 `CROSS_MODAL_LINK` 的使用频率不高。

## 8. `COMPARE_AGGREGATE` 做什么

代码位置：

- `_run_compare()`

它有两种模式。

### 8.1 候选路径聚合

如果依赖节点里有：

- `candidate_text_answers`
- `visual_relation_attempts`

它会构建 `candidate_paths`，每条包含：

- `candidate_entity`
- `candidate_image_id`
- `answer`
- `confidence`
- `selection_score`
- `evidence_hit`
- `query`

然后按：

```text
evidence_hit
selection_score
confidence
```

选择最高的一条，输出：

```json
{
  "type": "compare_candidate_paths",
  "selected_path": {...},
  "paths": [...]
}
```

### 8.2 普通比较

如果没有候选路径，它会退化为：

- 从依赖答案中抽数字；
- 如果问题问 higher/lower/largest/smallest，就比较数字；
- 如果是多个答案是否相同，就输出 yes/no；
- 否则拼接依赖答案。

### 8.3 当前问题

现在的 compare 还比较浅：

- 它不是完整 path-level reasoning。
- 它不会重新执行视觉验证。
- 它主要在已有 attempts 中选分最高的答案。
- 如果上游 candidate_state 已经丢了 gold，它无法恢复。

## 9. `VISUAL_IDENTIFY` 和 `VISUAL_MATCH`

### 9.1 `VISUAL_IDENTIFY`

语义：

> 用户真的给了 query image，现在需要看这张输入图。

如果有 `query_image_path`，它调用 VLM：

```text
visual_grounding_subquery(original_query, visual_sub_query, query_image)
```

如果没有 query image，它会 fallback 到 `IMAGE_RETRIEVE`。

所以在 MMQA 无问题图像的场景里，正常不应该大量使用 `VISUAL_IDENTIFY`。

### 9.2 `VISUAL_MATCH`

语义：

> 用用户提供的 query image 去 KB 里找相似图。

如果没有 query image，它也会 fallback 到 `IMAGE_RETRIEVE`。

对于当前 MMQA 外部图像问题，它不是主力。

## 10. KG 在 CoR 中到底起什么作用

当前 KG 的作用可以分成四类。

### 10.1 文本检索增强

`TEXT_RETRIEVE` 不是只拿 BGE chunk 检索。它会：

```text
BGE text/entity/relation retrieval
 -> KG beam search
 -> path-aware context
 -> LLM answer
```

这使得文本 hop 可以利用实体关系和多跳路径，而不是只看 top-k chunk。

### 10.2 图像候选映射成实体

`IMAGE_RETRIEVE` 检到的是 image id / image_path。但回答问题通常需要实体，比如：

```text
img-xxx -> Amanat Baghdad
img-yyy -> The Polar Express
```

KG 提供：

```text
image node -> visual entity -> text entity
image node -> page title/entity
```

从而让图片候选变成下游可用的文本实体。

### 10.3 图像候选重排增强

图像候选的 rerank 文本不仅包含 caption，还包含 KG path：

```text
linked_entities
caption
global_description
OCR
graph path src/tgt/description
```

reranker 因此能利用 KG 上的实体和关系信息，而不是只靠图像相似度。

### 10.4 跨模态路径证据

最终输出里每个 node 可以带：

- `path`
- `kg_context.graph_paths`
- `candidate_path_summary`

这些是最终 answer prompt 能看到的 KG/path 证据。

但是当前 final prompt 仍然是一次 LLM 综合，不是程序化地从 path table 中严格选择答案。

## 11. 为什么现在 KG 的好处还不明显

从最新 469 trace 看，问题不在于 KG 完全没用，而是 KG 用法还不够“约束化”。

### 11.1 KG 更多用于候选补信息，不是用于强制路径搜索

当前图像检索后 KG 做的是：

```text
image -> linked_entities
image -> graph_paths
```

它增强了候选描述和实体映射，但没有强制要求：

```text
必须存在 image -> visual_relation(open book) -> school -> table evidence
```

所以它还是偏 evidence enrichment，不是严格 symbolic constraint solving。

### 11.2 视觉关系没有充分执行

`VISUAL_RELATION` 只有 9 次，说明大多数视觉条件没有经过 VLM 验证。

例如：

- `open book logo`
- `stripe above and below eye`
- `white building with domed roof`
- `lion logo`
- `flying horse logo`

多数时候仍由 `IMAGE_RETRIEVE` 的 CLIP/rerank 处理。

这会让 KG 的视觉关系边和 VLM 视觉判断都没有充分发挥作用。

### 11.3 Planner 仍可能把约束顺序分错

比如 S. Cole 题，正确顺序应该是：

```text
TEXT_RETRIEVE: S. Cole played Muriel -> TV title
IMAGE_RETRIEVE / VISUAL_RELATION: inspect that title cover -> action
```

但现在可能规划成：

```text
IMAGE_RETRIEVE: retrieve TV title cover where S. Cole played Muriel
IMAGE_RETRIEVE: identify what person is doing in selected wrong image
```

这会让图像检索承担文本条件，导致第一步就跑偏。

### 11.4 下游候选验证仍是局部最优

`TEXT_RETRIEVE` 会验证多个候选，但选择逻辑还是：

```text
非 unknown + evidence_hit + selection_score
```

它不知道哪个候选满足原问题的全部约束。

例如 Vandit Bhatt 题：

- 正确答案需要先找到 poster 对应的 `Madam Secretary`。
- 但图像 top-5 没有它。
- 文本验证只能在错误候选里找 Vandit Bhatt 的角色。
- 最后选了 `The Fosters -> Declan Rivers`。

### 11.5 Final answer 不是严格 path selector

最终 prompt 会看到 `candidate_path_summary`，但仍然交给 LLM 自由生成。

这意味着：

- 如果 selected node answer 错了，final 可能继续沿用错答案。
- 如果 candidate_path_summary 中有正确候选但排序低，final 不一定会救回来。
- 如果候选路径结构不完整，final 没法程序化判定。

## 12. 用具体问题看当前算子链

### 12.1 Nona Gaye / train poster

问题：

```text
What was the role in Nona Gaye films for the title that features a train on the poster?
```

当前 CoR 大致执行：

1. `TEXT_RETRIEVE`

   查 Nona Gaye films，得到多个电影候选。

2. `IMAGE_RETRIEVE`

   在这些电影候选中找 poster featuring train。

   candidate_state 中 top 候选包括：

   - `The Polar Express`
   - `Harlem Nights`
   - `Blood and Bone`

   KG 把 top image 映射到：

   - `The Polar Express`
   - `The Polar Express (film)`

3. `TEXT_RETRIEVE`

   查询 Nona Gaye 在 `The Polar Express` 中的角色。

最终预测：

```text
Hero Girl (voice), also motion-capture
```

Gold：

```text
Hero Girl (voice)
```

分析：

- 这里路径基本正确。
- 错误主要是 final answer 粒度多带了 `also motion-capture`。
- 说明 candidate path 有效，但答案抽取仍不够精确。

### 12.2 S. Cole / Muriel / cover action

问题：

```text
What is the person in the S. Cole TV title cover where she played Muriel doing?
```

当前 CoR 执行：

1. `IMAGE_RETRIEVE`

   query：

   ```text
   Retrieve the KB/corpus image of the TV title cover where S. Cole played Muriel.
   ```

   top 候选：

   - `Command & Conquer 3: Kane`
   - `Command & Conquer 3: Tiberium Wars`
   - `Bones (TV series)`
   - `Waiting for God (TV series)`
   - `Colony (TV series)`

2. `IMAGE_RETRIEVE`

   又基于错误 top1 问：

   ```text
   Identify what the person is doing in Command & Conquer 3: Kane.
   ```

最终预测：

```text
Command & Conquer 3: Kane
```

Gold：

```text
writing
```

分析：

- 第一步顺序错了。
- 应该先文本查 `S. Cole played Muriel -> TV title`。
- 然后再对该 title 的 cover 做视觉识别。
- 当前把文本约束混进图像检索，导致 CLIP 检到完全错误的图。

### 12.3 Thrush / stripe / binomial

问题：

```text
In the list of thrushes and chats birds of Wales, the common name of the bird that has a stripe below and above its eye has what binomial?
```

当前 CoR 执行：

1. `TEXT_RETRIEVE`

   得到候选鸟列表：

   ```text
   Swainson's thrush, Grey-cheeked thrush, Ring ouzel, Common blackbird,
   Eyebrowed thrush, Dusky thrush, Red-throated thrush, Fieldfare,
   Song thrush, Redwing, Mistle thrush, American robin
   ```

2. `IMAGE_RETRIEVE`

   query：

   ```text
   Among [候选鸟列表], retrieve KB images or visual descriptions of birds
   that have a stripe below and above their eye.
   ```

   top 候选：

   - `Dusky thrush`
   - `Swainson`
   - `Eyebrowed thrush`
   - `Ring ouzel`
   - `Common blackbird`

3. `TEXT_RETRIEVE`

   在候选上查 binomial，最后选：

   ```text
   Eyebrowed thrush -> Turdus obscurus
   ```

Gold：

```text
Turdus iliacus
```

分析：

- 第一步文本候选中包含正确候选 `Redwing`。
- 第二步图像候选阶段把 gold 丢了。
- 没有 `VISUAL_RELATION` 对候选鸟图逐张判断 eye stripe。
- 所以它不是后续文本检索错，而是视觉谓词过滤错。

### 12.4 Vandit Bhatt / domed roof poster

问题：

```text
What role did Vandit Bhatt play in the television show that has a poster featuring a white building with a domed roof?
```

当前 CoR 执行：

1. `IMAGE_RETRIEVE`

   query：

   ```text
   Retrieve the television show whose poster features a white building with a domed roof.
   ```

   top 候选：

   - `Silver Spoon, Sterling Shackles`
   - `Coronation Street`
   - `With or Without You (TV series)`
   - `The Fosters (2013 TV series)`
   - `H.I.T (TV series)`

2. `TEXT_RETRIEVE`

   对这些候选验证 Vandit Bhatt role。

   其中 `The Fosters` 有文本 evidence，于是选：

   ```text
   Declan Rivers
   ```

Gold：

```text
Tech
```

分析：

- 正确路径应该先文本找 Vandit Bhatt 的 TV title 候选。
- 然后对每个 TV title 的 poster 做 visual relation 判断。
- 当前先全库搜 poster visual phrase，正确 title `Madam Secretary` 没进 top-5。
- 下游只能在错误候选里找 role。

## 13. 当前设计的真实问题

现在 CoR 的方向没有错，但当前代码实现还不是理想中的 CoR。

理想中的 CoR 应该是：

```text
候选集合
 -> 文本约束过滤
 -> 图像检索
 -> VLM 视觉谓词验证
 -> KG/entity/path 绑定
 -> 后续文本/表格属性查询
 -> 全局候选路径选择
```

当前 CoR 更接近：

```text
planner 给几个 node
 -> 每个 node 做检索并选一个局部 top answer
 -> 下游在 top-k candidate 上做有限验证
 -> final LLM 综合
```

差距主要有五点：

1. **operation 不够 typed**

   `IMAGE_RETRIEVE` 同时承担：

   - 找图像；
   - 判断视觉属性；
   - 映射文本实体；
   - 有时还承担文本候选过滤。

   这导致它太重，容易把混合约束全丢给 CLIP。

2. **视觉谓词没有强制执行**

   `VISUAL_RELATION` 虽然实现了，但调用很少。

   视觉属性题应该强制：

   ```text
   candidate image set -> VLM visual verification
   ```

3. **KG 是增强信息，不是硬约束**

   当前 KG 会补 linked entities 和 graph paths，但不会强制证明：

   ```text
   image candidate satisfies visual predicate
   AND linked entity appears in textual/table evidence
   AND target attribute comes from that entity
   ```

4. **候选状态还不是完整 path state**

   当前有：

   - `image_entity_candidates`
   - `candidate_text_answers`
   - `visual_relation_attempts`
   - `compare_candidate_paths`

   但这些状态没有统一成一张全局 candidate path table。

5. **final answer 仍然依赖 LLM 自由综合**

   即使候选路径里有正确信息，最终答案仍可能被 selected node answer 或局部高分候选带偏。

## 14. 后续应该怎么改

下一步不应该只是继续改 prompt，而是要把 operation 和 candidate state 设计得更硬。

### 14.1 把视觉检索拆成两个阶段

现在：

```text
IMAGE_RETRIEVE = 检索图像 + 视觉判断 + 实体映射
```

建议改成：

```text
IMAGE_RETRIEVE = 只负责召回候选图像/候选实体，不做最终视觉判断
VISUAL_RELATION = 对候选图像逐张 VLM 验证视觉谓词
CROSS_MODAL_LINK = 把验证通过的图像映射到文本/KG 实体
```

### 14.2 Planner 输出 constraint schema

每个 node 不只输出 operation，还应该输出：

```json
{
  "text_constraints": [...],
  "visual_constraints": [...],
  "candidate_set_source": "...",
  "target_attribute": "...",
  "answer_type": "entity|number|role|boolean"
}
```

这样可以避免把整句混合约束送进 CLIP。

### 14.3 候选路径统一成表

建议维护全局：

```json
{
  "path_id": "...",
  "candidate_entity": "...",
  "image_id": "...",
  "visual_checks": [...],
  "text_checks": [...],
  "kg_paths": [...],
  "score": ...
}
```

每个算子只更新这张表，而不是输出一个局部 answer。

### 14.4 KG 约束化

KG 不只是补描述，而应参与判断：

```text
image node must link to candidate entity
candidate entity must appear in text/table evidence
visual relation must be verified by VLM or KG visual edge
target answer must be read from same entity path
```

### 14.5 Final answer 从 selected path 生成

最终答案不应该让 LLM 在所有 node answer 中自由猜，而应该：

```text
selected_path -> answer extraction
```

LLM 可以做语言归一化，但不能重新选择无证据路径。

## 15. 一句话总结

当前 CoR 已经有了多模态算子、KG 图像实体映射、candidate_state 和候选验证的雏形；但是它的执行仍然偏“节点级局部答案链”，而不是“全局候选路径逐步约束”。KG 目前主要增强检索和实体映射，还没有成为强约束的跨模态推理路径。因此现在看起来渐进式和 KG 的好处不明显。

真正要把 CoR 做出优势，需要把 `IMAGE_RETRIEVE`、`VISUAL_RELATION`、`TEXT_RETRIEVE`、`COMPARE_AGGREGATE` 都改成围绕同一张 candidate path table 工作，让文本约束、视觉验证和 KG 路径共同决定最终候选。
