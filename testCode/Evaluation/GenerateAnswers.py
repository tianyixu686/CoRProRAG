import argparse
import csv
import json
import os
import re
import string
import sys
import traceback
import asyncio
from datetime import datetime
from tqdm import tqdm
import httpx
from openai import AsyncOpenAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.App import App
from src.config import settings
from src.prompt import PROMPTS
from src.utils import compute_exact_match, compute_f1


_judge_client = None


def get_judge_client():
    global _judge_client
    base_url = os.environ.get("EVAL_LLM_BASE_URL")
    api_key = os.environ.get("EVAL_LLM_API_KEY")
    if not base_url or not api_key:
        return None
    if _judge_client is None:
        _judge_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.AsyncClient(trust_env=False),
        )
    return _judge_client


def normalize_method(method: str) -> str:
    if method.strip().lower() in {"cor_simple", "simple_cor", "cor-simple"}:
        return "cor_simple"
    raise ValueError("Unsupported method. Available value: cor_simple (aliases: simple_cor, cor-simple).")


def build_query_text(q_obj, use_query_image_context=False):
    question = q_obj.get("question", "")
    if not use_query_image_context:
        return question
    metadata = q_obj.get("metadata", {}) or {}
    image_context = metadata.get("query_image_context", "")
    if not image_context:
        return question
    return f"{image_context}\nQuestion: {question}"


# ---------------------------- Single Sample Evaluation ----------------------------


def normalize_m3vqa_answer(text):
    """Normalize string answers for M3-VQA alias matching."""
    text = str(text).lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_answer_text(answer_obj):
    """Extract the direct answer string from known response shapes."""
    answer = answer_obj
    if isinstance(answer_obj, dict):
        answer = None
        if isinstance(answer_obj.get("response"), dict):
            answer = answer_obj["response"].get("answer")
        if answer is None:
            answer = answer_obj.get("answer")
        if answer is None:
            answer = answer_obj.get("response")
    return "" if answer is None else answer


def split_m3vqa_prediction(answer):
    """Split concise generated answers into a prediction set."""
    if isinstance(answer, list):
        values = []
        for item in answer:
            values.extend(split_m3vqa_prediction(item))
        return values

    text = str(answer).strip()
    text = re.sub(r"^(answer|final answer)\s*[:：]\s*", "", text, flags=re.I)
    parts = re.split(r"\s*\|\s*|\s*;\s*|\s*,\s*|\s*\n\s*", text)

    values = []
    seen = set()
    for part in parts:
        part = part.strip(" .;，、")
        norm = normalize_m3vqa_answer(part)
        if norm and norm not in seen:
            seen.add(norm)
            values.append(part)
    return values


def m3vqa_gold_groups(item):
    metadata = item.get("metadata", {}) or {}
    groups = metadata.get("answer_evals")
    if not groups:
        groups = [[answer] for answer in item.get("real_answer", [])]

    normalized_groups = []
    for group in groups:
        if not isinstance(group, list):
            group = [group]
        aliases = []
        seen = set()
        for alias in group:
            norm = normalize_m3vqa_answer(alias)
            if norm and norm not in seen:
                seen.add(norm)
                aliases.append(str(alias))
        if aliases:
            normalized_groups.append(aliases)
    return normalized_groups


def m3vqa_alias_match(prediction, aliases):
    pred_norm = normalize_m3vqa_answer(prediction)
    if not pred_norm:
        return False

    for alias in aliases:
        alias_norm = normalize_m3vqa_answer(alias)
        if not alias_norm:
            continue
        if pred_norm == alias_norm:
            return True
        if len(alias_norm) >= 4 and (alias_norm in pred_norm or pred_norm in alias_norm):
            return True

        pred_years = re.findall(r"\b\d{4}\b", pred_norm)
        alias_years = re.findall(r"\b\d{4}\b", alias_norm)
        if (
            len(pred_years) == 1
            and len(alias_years) == 1
            and abs(int(pred_years[0]) - int(alias_years[0])) <= 1
        ):
            return True
    return False


def compute_m3vqa_iou_accuracy(prediction, gold_groups):
    """Compute the M3-VQA answer-set IoU accuracy with alias groups."""
    pred_set = split_m3vqa_prediction(prediction)
    if not pred_set or not gold_groups:
        return 0.0

    adjacency = [
        [idx for idx, aliases in enumerate(gold_groups) if m3vqa_alias_match(pred, aliases)]
        for pred in pred_set
    ]
    matched_to = {}

    def dfs(pred_idx, seen):
        for gold_idx in adjacency[pred_idx]:
            if gold_idx in seen:
                continue
            seen.add(gold_idx)
            if gold_idx not in matched_to or dfs(matched_to[gold_idx], seen):
                matched_to[gold_idx] = pred_idx
                return True
        return False

    match_count = 0
    for pred_idx in range(len(pred_set)):
        if dfs(pred_idx, set()):
            match_count += 1

    denominator = len(pred_set) + len(gold_groups) - match_count
    return match_count / denominator if denominator else 0.0


from baselines.common.mmqa_metrics import score_answer


def evaluate_traditional_metrics(item):
    """Compute EM, F1, Precision, Recall for a single sample."""
    pred_answer = item["answer"]
    gold_answers = item["real_answer"]

    em, f1, precision, recall = score_answer(pred_answer, gold_answers, item.get("metadata", {}))
    best_metrics = dict(em=em, f1=f1, precision=precision, recall=recall)

    return {
        "query": item["query"],
        "prediction": pred_answer,
        "ground_truth": gold_answers,
        "EM": best_metrics["em"],
        "F1": best_metrics["f1"],
        "Precision": best_metrics["precision"],
        "Recall": best_metrics["recall"],
        "M3VQA_Accuracy": compute_m3vqa_iou_accuracy(
            pred_answer, m3vqa_gold_groups(item)
        )
        if (item.get("metadata", {}) or {}).get("dataset") == "m3vqa"
        else "",
    }


async def evaluate_with_llm(app, item):
    """Use LLM to judge whether the predicted answer is correct."""
    metadata = item.get("metadata", {}) or {}
    evidence = metadata.get("evidence", "")
    prompt_template = PROMPTS["eval_docbench_answer"] if evidence else PROMPTS["eval_answer"]
    prompt = prompt_template.format(
        question=item["query"],
        answer=item["answer"],
        gold_answer=item["real_answer"],
        evidence=evidence,
    )
    judge_model = os.environ.get("EVAL_LLM_MODEL_NAME", settings.llm_model_name)
    judge_client = get_judge_client()
    if judge_client is not None:
        response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()
    return await app.llm_model_func(prompt=prompt, model=judge_model)


def write_eval_result_to_csv(csv_file, traditional_metrics, llm_result):
    """Append evaluation result to CSV file."""
    row = traditional_metrics.copy()
    if isinstance(row["ground_truth"], list):
        row["ground_truth"] = "|".join(map(str, row["ground_truth"]))
    row["acc"] = llm_result

    fields = ["query", "prediction", "ground_truth", "EM", "F1", "Precision", "Recall", "M3VQA_Accuracy", "acc"]
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


def calculate_acc_ratio(csv_file):
    """Compute the ratio of LLM-judged correct answers in a CSV file."""
    try:
        correct_count = 0
        total_count = 0

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                if row.get("acc", "").strip().lower() == "correct":
                    correct_count += 1

        if total_count == 0:
            print("No data rows in CSV file")
            return 0.0

        acc_ratio = correct_count / total_count
        print("\nACC Results:")
        print(f"Total samples: {total_count}")
        print(f"Correct samples: {correct_count}")
        print(f"ACC (Accuracy): {acc_ratio:.4f} ({acc_ratio * 100:.2f}%)")

        return acc_ratio

    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
        return 0.0
    except Exception as e:
        print(f"Error computing ACC: {e}")
        return 0.0


# ---------------------------- Batch Evaluation ----------------------------


def evaluate_json_data(data):
    """Evaluate all samples in JSON data, returning aggregate metrics."""
    total = len(data)
    exact_match = 0
    f1_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    m3vqa_accuracy_total = 0.0
    m3vqa_count = 0
    sample_results = []

    for item in data:
        # Extract predicted answer from JSON structure
        pred_answer = extract_answer_text(item["answer"])
        gold_answer = item["real_answer"]

        best_em, best_f1, best_precision, best_recall = score_answer(
            pred_answer, gold_answer, item.get("metadata", {}))

        exact_match += best_em
        f1_total += best_f1
        precision_total += best_precision
        recall_total += best_recall
        m3vqa_accuracy = ""
        if (item.get("metadata", {}) or {}).get("dataset") == "m3vqa":
            m3vqa_accuracy = compute_m3vqa_iou_accuracy(
                pred_answer, m3vqa_gold_groups(item)
            )
            m3vqa_accuracy_total += m3vqa_accuracy
            m3vqa_count += 1

        sample_results.append(
            {
                "query": item["query"],
                "prediction": pred_answer,
                "ground_truth": gold_answer,
                "EM": best_em,
                "F1": best_f1,
                "Precision": best_precision,
                "Recall": best_recall,
                "M3VQA_Accuracy": m3vqa_accuracy,
            }
        )

    # Compute average scores (percentage)
    exact_match = 100.0 * exact_match / total if total else 0.0
    f1 = 100.0 * f1_total / total if total else 0.0
    precision = 100.0 * precision_total / total if total else 0.0
    recall = 100.0 * recall_total / total if total else 0.0
    m3vqa_accuracy = (
        100.0 * m3vqa_accuracy_total / m3vqa_count if m3vqa_count else None
    )

    return {
        "exact_match": exact_match,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "m3vqa_accuracy": m3vqa_accuracy,
        "m3vqa_count": m3vqa_count,
        "total": total,
        "sample_results": sample_results,
    }


def eval_and_print(input_file):
    """Load data from file, evaluate, and print results."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = evaluate_json_data(data)

    print("Overall Evaluation Results:")
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print(f"F1 Score: {results['f1']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall: {results['recall']:.2f}%")
    if results.get("m3vqa_accuracy") is not None:
        print(f"M3VQA Accuracy: {results['m3vqa_accuracy']:.2f}%")
    print(f"Total samples: {results['total']}")

    return results


def write_log_file(log_file, log_data):
    """Write log data to a JSON file."""
    try:
        log_entry = {"timestamp": datetime.now().isoformat(), **log_data}
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=4))
        print(f"Log saved to {log_file}")
    except Exception as e:
        print(f"Error writing log file: {e}")


async def fetch_single_query(
    app,
    q_obj,
    idx,
    method,
    use_query_image_context=False,
    use_query_image=False,
    skip_llm_judge=False,
    max_retries=3,
    retries=0,
):
    """Process a single query with retries."""
    try:
        print(f"Processing query {idx + 1}", flush=True)

        query_text = build_query_text(q_obj, use_query_image_context)
        metadata = q_obj.get("metadata", {}) or {}
        query_image_path = metadata.get("query_image_path") if use_query_image else None
        if query_image_path and not os.path.exists(query_image_path):
            raise FileNotFoundError(f"query image not found: {query_image_path}")
        print(f"Calling app.query for idx={idx}", flush=True)
        res = await app.query(query_text, method=method, query_image_path=query_image_path)
        print(f"app.query returned for idx={idx}", flush=True)

        answer = res.get("answer") if isinstance(res, dict) else res
        if answer is None and isinstance(res, dict):
            response = res.get("response")
            if isinstance(response, dict):
                answer = response.get("answer") or response.get("response")
            else:
                answer = response
        if answer is None:
            answer = "I don't know."
            if isinstance(res, dict):
                res = dict(res)
                res["answer"] = answer
                res.setdefault(
                    "structured_output_error",
                    "Model response omitted the required answer field.",
                )

        item = {
            "query": query_text,
            "answer": answer,
            "real_answer": q_obj["answer"],
            "metadata": q_obj.get("metadata", {}) or {},
        }

        eval1 = evaluate_traditional_metrics(item)
        eval2 = "Skipped" if skip_llm_judge else await evaluate_with_llm(app, item)

        return {
            "index": idx,
            "query": query_text,
            "raw_question": q_obj.get("question", ""),
            "metadata": q_obj.get("metadata", {}),
            "answer_obj": res,
            "real_answer": q_obj["answer"],
            "traditional_metrics": eval1,
            "llm_result": eval2,
        }
    except Exception as e:
        if retries < max_retries:
            print(f"Error at idx={idx}, retry {retries + 1}/{max_retries}: {e}")
            traceback.print_exc()
            return await fetch_single_query(
                app,
                q_obj,
                idx,
                method,
                use_query_image_context,
                use_query_image,
                skip_llm_judge,
                max_retries,
                retries + 1,
            )

        print(f"Failed idx={idx} after {max_retries} retries.")
        return {"index": idx, "error": str(e)}


async def process_queries(
    app,
    questions,
    output_json,
    output_csv,
    method="cor_simple",
    concurrency=5,
    use_query_image_context=False,
    use_query_image=False,
    skip_llm_judge=False,
):
    """Process all queries asynchronously with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_fetch(i, q_obj):
        async with semaphore:
            return await fetch_single_query(
                app,
                q_obj,
                i,
                method,
                use_query_image_context=use_query_image_context,
                use_query_image=use_query_image,
                skip_llm_judge=skip_llm_judge,
            )

    tasks = [
        asyncio.create_task(_bounded_fetch(i, question))
        for i, question in enumerate(questions)
    ]

    total = len(tasks)
    outputs = []
    with tqdm(total=total, desc="Processing", unit="query") as pbar:
        for task in asyncio.as_completed(tasks):
            out = await task
            if "error" in out:
                q_obj = questions[out["index"]]
                item = {"query": q_obj["question"], "answer": "I don't know.",
                        "real_answer": q_obj["answer"], "metadata": q_obj.get("metadata", {})}
                out.update(query=item["query"], raw_question=item["query"], metadata=item["metadata"],
                           answer_obj={"answer": "I don't know.", "execution_error": out.pop("error")},
                           real_answer=item["real_answer"], traditional_metrics=evaluate_traditional_metrics(item),
                           llm_result="Skipped")
            outputs.append(out)
            # Each completed question survives interruption; index preserves dataset identity.
            with open(output_json + ".partial.jsonl", "a", encoding="utf-8") as checkpoint:
                checkpoint.write(json.dumps(out, ensure_ascii=False) + "\n")
            pbar.update(1)

    final_results = []
    for i, out in enumerate(sorted(outputs, key=lambda x: x.get("index", -1))):
        if out and "error" not in out:
            write_eval_result_to_csv(
                output_csv, out["traditional_metrics"], out["llm_result"]
            )
            final_results.append(
                {
                    "query": out["query"],
                    "raw_question": out.get("raw_question", out["query"]),
                    "metadata": out.get("metadata", {}),
                    "answer": out["answer_obj"],
                    "real_answer": out["real_answer"],
                }
            )
        else:
            print(f"Query at index {i} failed: {out.get('error', 'Unknown error')}")

    with open(output_json, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(final_results, ensure_ascii=False, indent=4))
    print(f"Saved {len(final_results)} records to {output_json}")

    return final_results


async def process_once(
    run_id,
    dataset_path,
    eval_count,
    concurrency,
    method="cor_simple",
    start_idx=0,
    output_json_file=None,
    output_csv_file=None,
    log_file=None,
    use_query_image_context=False,
    use_query_image=False,
    skip_llm_judge=False,
):
    """Single run: process questions in parallel and evaluate."""
    print(f"Run {run_id}: Initializing App...")
    app = App(settings)
    print(f"Run {run_id}: App initialized successfully.")

    try:
        with open(dataset_path, "r", encoding="utf-8") as file:
            questions = json.load(file)

        print(f"Run {run_id}: Loaded {len(questions)} questions from {dataset_path}")

        # Write CSV header
        fieldnames = [
            "query",
            "prediction",
            "ground_truth",
            "EM",
            "F1",
            "Precision",
            "Recall",
            "M3VQA_Accuracy",
            "acc",
        ]
        with open(output_csv_file, "w", encoding="utf-8") as f:
            f.write(",".join(fieldnames) + "\n")

        end_idx = len(questions) if eval_count is None else start_idx + eval_count
        selected_questions = questions[start_idx:end_idx]

        print(
            f"Run {run_id}: Processing {len(selected_questions)} questions (index {start_idx}-{start_idx + len(selected_questions) - 1}) with method '{method}', concurrency={concurrency}, use_query_image={use_query_image}, use_query_image_context={use_query_image_context}, skip_llm_judge={skip_llm_judge}..."
        )

        results = await process_queries(
            app=app,
            questions=selected_questions,
            output_json=output_json_file,
            output_csv=output_csv_file,
            method=method,
            concurrency=concurrency,
            use_query_image_context=use_query_image_context,
            use_query_image=use_query_image,
            skip_llm_judge=skip_llm_judge,
        )

        print(f"Run {run_id}: All answers generated. Evaluating output files...")
        eval_results = eval_and_print(output_json_file)

        print(f"Run {run_id} ACC results:")
        acc_ratio = calculate_acc_ratio(output_csv_file)

        result_data = {
            "run_id": run_id,
            "status": "success",
            "processed_questions": len(selected_questions),
            "successful_results": len(results),
            "use_query_image_context": use_query_image_context,
            "use_query_image": use_query_image,
            "skip_llm_judge": skip_llm_judge,
            "acc_ratio": acc_ratio,
            "exact_match": eval_results["exact_match"],
            "f1": eval_results["f1"],
            "precision": eval_results["precision"],
            "recall": eval_results["recall"],
            "m3vqa_accuracy": eval_results.get("m3vqa_accuracy"),
            "output_files": {"json": output_json_file, "csv": output_csv_file},
        }

        if log_file:
            write_log_file(log_file, result_data)

        return result_data

    except Exception as e:
        error_msg = f"Run {run_id} failed with error: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        error_data = {"run_id": run_id, "status": "error", "error": str(e)}
        if log_file:
            write_log_file(log_file, error_data)

        return error_data

    finally:
        try:
            await app.close_cache_only()
            print(f"Run {run_id}: Cache resources cleaned up.")
        except Exception as e:
            print(f"Run {run_id}: Error during cleanup: {e}")


def build_arg_parser():
    default_dataset = settings.root_dir + "/Input/dataset_for_hipporag2/sample.json"
    parser = argparse.ArgumentParser(
        description="Generate answers and run evaluation."
    )
    parser.add_argument(
        "--method",
        default="cor_simple",
        help="Query method: cor_simple",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=1,
        help="Number of questions to evaluate from start index.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent query/evaluation tasks.",
    )
    parser.add_argument(
        "--dataset-path",
        default=default_dataset,
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index in dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=settings.root_dir + "/Output",
        help="Output root directory.",
    )
    parser.add_argument(
        "--use-query-image-context",
        action="store_true",
        help="Append eval metadata.query_image_context to each question, useful for M3-VQA.",
    )
    parser.add_argument(
        "--use-query-image",
        action="store_true",
        help="Pass eval metadata.query_image_path to the multimodal RAG query path.",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip the extra LLM-as-judge call; use dataset-specific evaluators for final metrics.",
    )
    return parser


async def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.eval_count is not None and args.eval_count <= 0:
        raise ValueError("--eval-count must be positive.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive.")

    run_id = datetime.now().strftime("%m_%d_%H_%M")
    method = normalize_method(args.method)

    output_json_file = f"{args.output_dir}/Ans/predicted_answers_{run_id}.json"
    output_csv_file = f"{args.output_dir}/Eval/predicted_answers_{run_id}.csv"
    log_file = f"{args.output_dir}/Log/run_log_{run_id}.json"

    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    await process_once(
        run_id=run_id,
        dataset_path=args.dataset_path,
        eval_count=args.eval_count,
        concurrency=args.concurrency,
        method=method,
        start_idx=args.start_idx,
        output_json_file=output_json_file,
        output_csv_file=output_csv_file,
        log_file=log_file,
        use_query_image_context=args.use_query_image_context,
        use_query_image=args.use_query_image,
        skip_llm_judge=args.skip_llm_judge,
    )


if __name__ == "__main__":
    asyncio.run(main())
