# cor_simple development branch

This branch keeps **cor_simple** as the only CoR query implementation. It uses
natural-language question decomposition, shared multimodal SGN retrieval,
sub-question answering, the existing bounded corrector, and final synthesis.
The legacy typed-LEG and candidate-path/Logic Harness executors are removed.
Dataset preparation, indexing, external comparison baselines, and historical
results are retained; historical commands for retired CoR methods no longer run.

Use `--method cor_simple` with `testCode/Evaluation/GenerateAnswers.py`.
Regression checks (mock model calls; no GPU services required):

```bash
LLM_API_KEY=test-only EMBEDDING_API_KEY=test-only OPENAI_API_KEY=test-only \
  .conda_env/bin/python -m unittest testCode.test_cor_simple_alignment -v
```

Implementation fixes on this branch: infer placeholder dependencies, preserve
all compact visual observations, retain every step in final answer packing,
handle `wear eyeglasses` candidate filters, keep visible questions on the image
path, and never substitute an arbitrary intermediate answer for a failed final
answer. These are implementation corrections, not new research contributions.
No new MMQA score is claimed before rerunning the evaluation.

The original project documentation follows for historical context.

## Quick Start

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set API environment variables (example):

```bash
export LLM_BASE_URL="..."
export LLM_API_KEY="..."
export EMBEDDING_BASE_URL="..."
export EMBEDDING_API_KEY="..."
export RERANK_BASE_URL="..."
export RERANK_API_KEY="..."
```

## Build Knowledge Graph

Script: `testCode/Tools/kg_generate.py`

```bash
python testCode/Tools/kg_generate.py --input-path Input/dataset_for_hipporag2/sample_corpus.json
```

Parameter:

- `--input-path`: corpus JSON path to process.
- If you build KG for a different dataset, update `data_dir` in `src/config.py` first.

## Prepare Multimodal QA Data

Script: `testCode/Tools/prepare_multimodal_dataset.py`

The converter writes two files that match the existing pipeline:

- `*_corpus.json`: input for KG construction. Each record uses `{title, text, images?}`.
- `*.json`: input for answer generation/evaluation. Each record uses `{question, answer}`.

Recommended local layout:

```text
Input/
  raw/
    multimodalqa/
      MultiModalQA_dev.jsonl.gz
      texts.jsonl.gz
      tables.jsonl.gz
      images.jsonl.gz
      images/final_dataset_images/
    m3vqa/
      m3vqa_test.json
      images/
      kb/                       # optional controlled knowledge base
  dataset_for_multimodalqa/
  dataset_for_m3vqa/
```

### MultiModalQA

MultiModalQA contains questions plus three context files:

- `MultiModalQA_train/dev/test.jsonl.gz`: one QA example per line.
- `texts.jsonl.gz`: Wikipedia text paragraph contexts.
- `tables.jsonl.gz`: table contexts.
- `images.jsonl.gz`: image metadata. The actual image files are in `images.zip`.

Important fields:

- QA row: `qid`, `question`, `answers`, `metadata`, `supporting_context`.
- `metadata`: includes `modalities`, `text_doc_ids`, `image_doc_ids`, `table_id`, and intermediate answers.
- image row: `id`, `title`, `url`, `path`; `path` is relative to the unzipped image directory.

```bash
python testCode/Tools/prepare_multimodal_dataset.py \
  --dataset multimodalqa \
  --questions Input/raw/multimodalqa/MultiModalQA_dev.jsonl.gz \
  --texts Input/raw/multimodalqa/texts.jsonl.gz \
  --tables Input/raw/multimodalqa/tables.jsonl.gz \
  --images Input/raw/multimodalqa/images.jsonl.gz \
  --image-root Input/raw/multimodalqa/images/final_dataset_images \
  --output-dir Input/dataset_for_multimodalqa \
  --name dev \
  --limit 5
```

The `--limit 5` flag creates a tiny smoke-test subset first. Remove it for the
full split.

For the multimodal CoRProRAG benchmark, first select questions that always
combine image evidence with text/table evidence. In this selector, `text` and
`table` are both treated as textual evidence:

```bash
python testCode/Tools/select_multimodalqa_corprorag_subset.py \
  --questions Input/raw/multimodalqa/MultiModalQA_dev.jsonl.gz \
  --images Input/raw/multimodalqa/images.jsonl.gz \
  --image-root Input/raw/multimodalqa/images/final_dataset_images \
  --output-dir Input/dataset_for_multimodalqa_corprorag
```

Then convert each selected subset with the standard converter:

```bash
python testCode/Tools/prepare_multimodal_dataset.py \
  --dataset multimodalqa \
  --questions Input/dataset_for_multimodalqa_corprorag/mmqa_image_textual_smoke_questions.jsonl \
  --texts Input/raw/multimodalqa/texts.jsonl.gz \
  --tables Input/raw/multimodalqa/tables.jsonl.gz \
  --images Input/raw/multimodalqa/images.jsonl.gz \
  --image-root Input/raw/multimodalqa/images/final_dataset_images \
  --output-dir Input/dataset_for_multimodalqa_corprorag \
  --name mmqa_image_textual_smoke
```

Repeat with `mmqa_image_textual_main_questions.jsonl` and
`--name mmqa_image_textual_main` for the 200-question main split. The converted
eval JSON preserves `metadata.corprorag_selection` for grouped analysis.

Then build and evaluate the small subset:

```bash
export CORPRORAG_DATA_DIR=/path/to/indexes/mmqa_dev

python testCode/Tools/kg_generate.py \
  --input-path Input/dataset_for_multimodalqa/dev_corpus.json

python testCode/Evaluation/GenerateAnswers.py \
  --method corprorag \
  --dataset-path Input/dataset_for_multimodalqa/dev.json \
  --eval-count 50 \
  --concurrency 8
```

### M3-VQA

M3-VQA is a knowledge-based visual QA dataset for multi-entity, multi-hop
reasoning. The Hugging Face dataset has fields such as:

- `data_id`: sample id.
- `image_id`: filename for the question image.
- `question`, `question_type`, `question_hop`, `entity_num`.
- `answers`, `answer_evals`.
- `img_entity_names`: main entities in the image.
- `evidence`, `evidence_urls`, `evidence_img_ids`.
- `evidence_section_ids`, `evidence_section_titles`, `evidence_url_titles`.

The converter can run in three corpus modes:

- gold-evidence mode: use per-question `evidence` fields only.
- KB mode: pass `--knowledge-base` and build from the controlled KB as one shared retrieval corpus.
- both mode: combine the controlled KB with per-question gold evidence for debugging or upper-bound analysis.

When `--knowledge-base` is provided, `--m3vqa-corpus-mode auto` defaults to KB
mode. This keeps the main retrieval experiment faithful to the M3-VQA protocol:
all documents are indexed together first, then questions retrieve from that
shared corpus. Gold evidence is not mixed into the KB unless you explicitly set
`--m3vqa-corpus-mode both`.

```bash
python testCode/Tools/prepare_multimodal_dataset.py \
  --dataset m3vqa \
  --questions Input/raw/m3vqa/quesions.jsonl \
  --image-root Input/raw/m3vqa/images \
  --output-dir Input/dataset_for_m3vqa \
  --name test \
  --limit 5
```

`--knowledge-base` is optional. If omitted, the converter builds the corpus from
per-question evidence fields such as `evidence`, `evidence_urls`, and
`evidence_img_ids`.

For KB mode:

```bash
python testCode/Tools/prepare_multimodal_dataset.py \
  --dataset m3vqa \
  --questions Input/raw/m3vqa/quesions.jsonl \
  --knowledge-base Input/raw/m3vqa/kb/encyclopedic_kb_wiki.json \
  --m3vqa-corpus-mode kb \
  --m3vqa-kb-filter-to-question-urls \
  --image-root Input/raw/m3vqa/images \
  --output-dir Input/dataset_for_m3vqa \
  --name test_kb \
  --limit 5
```

`--m3vqa-kb-filter-to-question-urls` keeps a smoke run practical by streaming
only KB pages referenced by the selected questions. Omit it for a full
controlled-KB build, but expect a much larger KG/indexing job because the EVQA
KB JSON is about 15 GB after extraction.

M3-VQA questions include a query image. The current CoRProRAG query interface is
text-only, so the converter stores `metadata.query_image_context` as a controlled
text proxy based on `img_entity_names`. Use it during evaluation with:

```bash
python testCode/Evaluation/GenerateAnswers.py \
  --method corprorag \
  --dataset-path Input/dataset_for_m3vqa/test_kb.json \
  --use-query-image-context \
  --eval-count 50 \
  --concurrency 8
```

By default, question images are not indexed as corpus documents. If you need a
diagnostic run that indexes them, add `--include-m3vqa-question-images` during
conversion and keep that run separate from the main KB evaluation.

Use a separate `CORPRORAG_DATA_DIR` for each dataset/split so vector stores,
KV caches, and graph files do not mix across experiments.

#### M3-VQA small offline-KB subsets

For cost-controlled offline KG experiments, first materialize stratified subsets
and their own controlled KB files:

```bash
python testCode/Tools/prepare_m3vqa_subset.py \
  --questions Input/raw/m3vqa/quesions.jsonl \
  --kb-path Input/raw/m3vqa/kb/encyclopedic_kb_wiki.json \
  --output-dir Input/raw/m3vqa/subsets \
  --sizes 50 100 300 500 \
  --seed 42 \
  --extract-kb
```

This writes files such as:

```text
Input/raw/m3vqa/subsets/m3vqa_subset_50.jsonl
Input/raw/m3vqa/subsets/m3vqa_subset_50_kb.json
Input/raw/m3vqa/subsets/m3vqa_subset_stats.json
```

Then convert a subset into the normal CoRProRAG corpus/eval format:

```bash
python testCode/Tools/prepare_multimodal_dataset.py \
  --dataset m3vqa \
  --questions Input/raw/m3vqa/subsets/m3vqa_subset_50.jsonl \
  --knowledge-base Input/raw/m3vqa/subsets/m3vqa_subset_50_kb.json \
  --m3vqa-corpus-mode kb \
  --image-root Input/raw/m3vqa/images \
  --output-dir Input/dataset_for_m3vqa \
  --name m3vqa_subset_50
```

Prepared subset-50 and subset-500 files are available under
`Input/dataset_for_m3vqa/`. For resource-limited experiments, use the minimal
baseline Slurm entrypoint:

```bash
sbatch scripts/sbatch_run_minimal_multimodal_baselines.sh
```

Override `DATASET_NAME=m3vqa_subset_500` when moving beyond the smoke split.

M3-VQA prediction files can be evaluated with answer aliases and grouped
breakdowns:

```bash
python testCode/Evaluation/evaluate_m3vqa.py \
  --predictions Output/Ans/predicted_answers_<run_id>.json \
  --output-json Output/Eval/m3vqa_<run_id>_report.json \
  --output-csv Output/Eval/m3vqa_<run_id>_samples.csv
```

## Generate Answers + Evaluate

Script: `testCode/Evaluation/GenerateAnswers.py`

```bash
python testCode/Evaluation/GenerateAnswers.py \
  --method corprorag \
  --eval-count 50 \
  --concurrency 8 \
  --dataset-path Input/dataset_for_hipporag2/sample.json
```

Parameters:

- `--method`: `corprorag` or `SGN` (compatible aliases: `logic_graph`, `beam_search_kg`).
- `--eval-count`: number of samples to evaluate.
- `--concurrency`: async parallel task count.
- `--dataset-path`: QA dataset JSON path(directory ForQA).
- `--start-idx` (optional): start index in dataset.
- `--output-dir` (optional): output directory, default `Output`.
- If you evaluate a different dataset, also update `data_dir` in `src/config.py` to the corresponding database path.

Output files are written to:

- `Output/Ans/*.json`
- `Output/Eval/*.csv`
- `Output/Log/*.json`
