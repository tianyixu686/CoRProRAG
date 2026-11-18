# MM-CoRProRAG 核心消融实验计划

## 目的与版本

按用户指定的四个版本，检验逐跳问题分解、SGN 检索和自适应修正器（AC）对 MMQA 的贡献。三个新变体使用当前 `codex/cor-simple-clean` 分支作为实现基底；每次运行记录 Git commit、配置、数据与索引哈希。按用户决定，不再重复 Full 的 469 题运行；Full 使用旧任务 61396 的已有结果，报告中明确其历史代码版本与新变体版本不同。

## 四个实验版本

| 标识 | 定义 | 保留 | 移除/替换 | 主比较 |
|---|---|---|---|---|
| `sgn_only` | 不调用分解 planner，直接把原题作为唯一问题送入当前 SGN 检索与回答流程 | 文本 SGN 的候选融合、KG beam search；当前图像分支的检索、KG/实体上下文、视觉处理；同一最终回答提示与证据预算 | 逐跳计划、跨跳答案传递、依赖纠错 | `full` vs `sgn_only`：逐跳分解的整体贡献 |
| `without_sgn` | 生成并执行与 Full 相同的子问题计划，但每个子问题用普通文本/图像检索代替 SGN 检索 | 分解 planner、策略选择、依赖答案传递、子问题回答、视觉验证、AC、最终原题证据检索与合成 | 文本 SGN 的 KG/来源融合/beam；图像 SGN 的实体/KG 上下文融合；用同库中的普通文本/图像向量候选替换 | `full` vs `without_sgn`：SGN 检索整体贡献 |
| `without_ac` | 完整流程，仅将 `cor_simple_max_correction_depth=0` | 其他所有步骤及预算 | 自适应修正和递归分解 | `full` vs `without_ac`：AC 的贡献 |
| `full` | 当前 `cor_simple` 默认配置 | 全部步骤 | 无 | 参照组 |

`sgn_only` 是独立 SGN 方法：原题只执行一次检索/回答，然后进行一次最终合成；不得因复用当前 `cor_simple_query` 的末尾逻辑而重复原题 SGN 检索。无分解时 AC 没有子问题失败可修正，因此不人为调用它。当前图像分支没有 KG beam search，不能把 `sgn_only` 解释为图像路径探索消融。

`without_sgn` 不是已有的 `sequential_mmrag` baseline。后者有独立提示词、证据打包和回答程序。新版本必须复用 Full 的 planner、策略决策、跨跳传递、视觉验证与回答/最终合成代码，仅通过一个检索后端开关替换文本和图像 SGN 检索。普通检索使用同一共享库的文本块向量索引和图像向量索引；不读取 KG 实体/关系/图像关联信息。最终原题的 SGN 检索也使用相同后端替换，否则仍残留 SGN。保留 Full 的候选/上下文预算，记录各分支实际取回数量；不以不同方法的 top-k 数字相等冒充计算等价。

## 实现边界

1. 在查询入口加入显式 `ablation_variant` 枚举，默认 `full`；将其写入逐题结果及运行 manifest。非法值报错。
2. 把当前 `cor_simple_query` 的 planner、逐跳执行、最终回答拆成可复用阶段。`sgn_only` 跳过 planner 和逐跳循环，只对原题执行一次；最终合成复用同一提示模板与 token 上限，不再执行第二次检索。
3. 为 `_simple_text_retrieve` 和 `_simple_image_retrieve` 增加可替换的普通检索后端；只在 `without_sgn` 调用。保持下游证据结构和答题逻辑一致。普通后端不能间接调用 `fuse_kg_text`、`beam_search` 或图像 KG 上下文。
4. `without_ac` 优先复用现有 `COR_SIMPLE_MAX_CORRECTION_DEPTH=0`，不要额外改答题提示或失败兜底。
5. 统一逐题记录 planner/Qwen/VLM 调用、输入输出 token、图像输入数、文本/图像检索与重排调用、阶段耗时、总在线延迟和异常。GPT-4o-mini API 费用按实际计费 token 计算；本地 Qwen 报 GPU 秒/题而非 0 美元。离线 KG/图像描述/索引成本另列。

## 数据、评测和运行

- 使用 `experiments/configs/mmqa_cor_simple_current469.json` 固定的 469 题和共享索引；所有题保留在分母，错误和超时单列。该集曾用于诊断，不标为独立未见测试集。
- 三个新变体在同一 commit、同一 Qwen3-VL-32B 服务、相同解码参数、并发和缓存策略下运行；历史 Full 使用其原始运行配置。Qwen 部署只在 GPU4；其他任务可用 GPU0/1。
- 历史 Full 的 469 份 `simple_plan` 已与当前 469 题逐题核对，均可直接重放且标准化结果不变。`without_sgn`、`without_ac` 的正式全量运行重放这些计划，避免新的 planner 输出波动；`sgn_only` 不调用 planner。每题仍由 Qwen 正常生成，模型回答不重放。
- 先完成已提交的四版 8 题 smoke，以验证各路径、调用数及无 SGN 的无 KG 访问；随后仅跑 `sgn_only`、`without_sgn`、`without_ac` 的 469 题。小样本只作实现检查。
- 统一重算本项目 `local_mmqa_set_v1_not_official` EM/F1，所有 469 题保留在分母；另报每题成功率、配对题目变化及题型/模态差异。Full 使用 61396 的已有输出，但由于代码版本不同，Full 与三个新变体的差值属于跨版本比较，不能表述为严格同代码版本的单因素因果消融。

## 输出表

主消融表：`Variant | EM | F1 | 成功率 | ΔF1 vs Full`。

效率表：`Variant | GPT-4o-mini calls/tokens | Qwen calls/input/output tokens | images/query | retrieval/rerank calls | latency mean/P95 | API USD/query | local GPU-sec/query`。token 按模型分列，不能把 VLM 图像数隐藏在文本 token 中。在线延迟从提交问题到最终答案，包含检索与所有模型调用，不含部署及建库。另表记录离线建库时间、GPU 时和存储。

补充机制分析：比较 `full` 与 `sgn_only` 在依赖型题上的差异；比较 `full` 与 `without_sgn` 的文本/图像支持证据命中率；比较 `full` 与 `without_ac` 在实际触发纠错的题及未触发题上的差异。支持证据指标须先核对标注与索引 ID 的映射，不使用无法可靠映射的样本计算召回。

## 完成判据

四版 8 题 smoke 均可从同一入口重现；三个新变体逐题输出包含配置身份；无 SGN 路径经调用追踪确认不访问 KG；SGN-only 不重复检索原题；469 道题完整计入三个新变体的指标。历史 Full 与新变体的代码版本差异须随主表披露。

## 当前实现与测试入口

运行变体由环境变量 `COR_SIMPLE_ABLATION_VARIANT` 控制，合法值为 `full`、`sgn_only`、`without_sgn`、`without_ac`。统一提交器为 `scripts/submit_mmqa_core_ablation.py`，读取冻结的 469 题 manifest，校验数据/索引哈希，并为每次运行写出变体、代码哈希和模型配置。它不保存或打印 planner 密钥；实际提交前需在环境中提供 `SIMPLE_PLANNER_API_KEY`（SGN-only 不需要）。

先运行少题 smoke：

```bash
.conda_env/bin/python scripts/submit_mmqa_core_ablation.py --variant full --count 8 --run-id core_full_smoke8 --dry-run
.conda_env/bin/python scripts/submit_mmqa_core_ablation.py --variant sgn_only --count 8 --run-id core_sgn_only_smoke8 --dry-run
.conda_env/bin/python scripts/submit_mmqa_core_ablation.py --variant without_sgn --count 8 --run-id core_without_sgn_smoke8 --dry-run
.conda_env/bin/python scripts/submit_mmqa_core_ablation.py --variant without_ac --count 8 --run-id core_without_ac_smoke8 --dry-run
```

四版 8 题 smoke 已提交。全量只提交三个新变体，使用 `--count 469` 和新的 run ID；不提交 Full 全量。在线成本精确统计需要另行打通模型返回的原始 usage 与 Qwen 服务 GPU 用量，不能用 prompt 长度估算替代实际账单/用量。历史 Full 若没有同口径资源日志，效率表对应单元格标记为未测，不能用其他运行推算。

正式全量的两个分解变体使用 `--plan-replay Output/runs/multimodalqa/corprorag_fullkg_sanity/cor_simple_full469_fixed_0922/Ans/predicted_answers_09_22_19_30.json`。重放运行不调用在线 planner，因此其计时与调用数不能直接作为自然端到端效率数值；正式效率比较需另行测量或明确扣除此差异。
