#!/usr/bin/env python3
"""Submit one core MM-CoRProRAG ablation on the frozen MMQA subset."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full", "sgn_only", "without_sgn", "without_ac")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node", choices=("GPU0", "GPU1"), default="GPU1")
    parser.add_argument("--mem-gb", type=int, default=64)
    parser.add_argument("--time-limit", default="0-16:00:00")
    parser.add_argument("--dependency", help="Slurm dependency, e.g. after:12345+10 or afterok:12345")
    parser.add_argument("--plan-replay", type=Path, help="Saved Full answers containing one plan per question")
    parser.add_argument("--plain-image-k", type=int, choices=(4, 5), default=5,
                        help="Image top-k for the plain w/o SGN execution path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    frozen = json.loads((ROOT / "experiments/configs/mmqa_cor_simple_current469.json").read_text())
    if args.start < 0 or args.count < 1 or args.start + args.count > frozen["count"]:
        parser.error("question range is outside the frozen 469-question set")
    if args.mem_gb < 16:
        parser.error("memory must be at least 16 GB")
    for relative, expected in frozen["sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen input changed: {relative}")
    output = ROOT / "Output/runs/multimodalqa/core_ablation" / args.run_id
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Fresh run ID required: {output}")
    if args.plan_replay and args.variant == "sgn_only":
        parser.error("SGN-only has no decomposition plan to replay")
    if args.plan_replay and not args.plan_replay.is_file():
        parser.error("plan replay file does not exist")
    env = os.environ.copy()
    needs_planner_key = args.variant in {"full", "without_sgn"} or (args.variant == "without_ac" and not args.plan_replay)
    if needs_planner_key and not (env.get("SIMPLE_PLANNER_API_KEY") or env.get("PLANNER_API_KEY")) and not args.dry_run:
        raise RuntimeError("Set SIMPLE_PLANNER_API_KEY for decomposition or adaptive correction")
    env.update({
        "COR_SIMPLE_ABLATION_VARIANT": args.variant,
        "COR_SIMPLE_PLAN_REPLAY_PATH": str(args.plan_replay.resolve()) if args.plan_replay else "",
        "COR_SIMPLE_MAX_CORRECTION_DEPTH": "0" if args.variant == "without_ac" else "1",
        "METHOD": "cor_simple", "QUESTIONS_PATH": frozen["questions"],
        "CORPRORAG_DATA_DIR": frozen["index"], "EVAL_COUNT": str(args.count),
        "START_IDX": str(args.start), "CONCURRENCY": "1", "RUN_ID": args.run_id,
        "OUTPUT_DIR": str(output), "USE_VLM_AS_LLM": "1",
        "VLM_BASE_URL": "http://GPU4:18000/v1", "VLM_MODEL_NAME": frozen["model"],
        "SIMPLE_PLANNER_BASE_URL": env.get("SIMPLE_PLANNER_BASE_URL", "https://api.apiyi.com/v1"),
        "SIMPLE_PLANNER_MODEL_NAME": "gpt-4o-mini",
        "WAIT_EXTERNAL_MODEL_SECONDS": "3600",
        "NO_PROXY": "localhost,127.0.0.1,GPU4,gpu4",
        "no_proxy": "localhost,127.0.0.1,GPU4,gpu4",
        "HTTPX_NO_PROXY": "localhost,127.0.0.1,GPU4,gpu4",
    })
    if args.variant == "without_sgn":
        env.update({"KG_TOP_K": "8", "IMAGE_RETRIEVAL_CANDIDATE_K": str(args.plain_image_k)})
    manifest = {
        "variant": args.variant, "count": args.count, "start": args.start,
        "questions": frozen["questions"], "index": frozen["index"], "output": str(output),
        "metric": frozen["metric"], "planner": "none" if args.variant == "sgn_only" else ("replayed_full_plan" if args.plan_replay else "gpt-4o-mini"),
        "adaptive_correction_api_enabled": args.variant in {"full", "without_sgn"},
        "retrieval_protocol": {"text": "dense_chunk_top8", "image": f"clip_image_top{args.plain_image_k}_raw_vlm_each"}
        if args.variant == "without_sgn" else "sgn",
        "plan_replay_path": str(args.plan_replay.resolve()) if args.plan_replay else None,
        "plan_replay_sha256": hashlib.sha256(args.plan_replay.read_bytes()).hexdigest() if args.plan_replay else None,
        "reader": frozen["model"], "reader_service": env["VLM_BASE_URL"],
        "dependency": args.dependency, "compute_node": args.node,
        "memory_gb": args.mem_gb, "time_limit": args.time_limit,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src/config.py", "src/pipeline/query_executor.py", "scripts/submit_mmqa_core_ablation.py"], cwd=ROOT, text=True).strip()),
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            ROOT / "src/config.py", ROOT / "src/pipeline/query_executor.py", ROOT / "scripts/submit_mmqa_core_ablation.py")},
        "input_sha256": frozen["sha256"],
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    output.mkdir(parents=True)
    (output / "ablation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    command = ["sbatch", "--parsable", "--nodelist=" + args.node,
               "--mem=" + str(args.mem_gb) + "G", "--time=" + args.time_limit]
    if args.dependency:
        command.append("--dependency=" + args.dependency)
    command.append(str(ROOT / "scripts/sbatch_run_mmqa_corprorag_sanity.sh"))
    job = subprocess.run(command, env=env, cwd=ROOT, check=True, text=True, capture_output=True)
    print("Slurm job ID: " + job.stdout.strip())


if __name__ == "__main__":
    main()
