#!/usr/bin/env python3
"""Checkpointed Microsoft GraphRAG Global Search evaluation on benchmark v2."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parent / "shared"
sys.path.insert(0, str(SHARED / "scripts"))
from benchmark_utils import read_jsonl, score_answer, summarize_results


def envfile(path: Path) -> dict[str, str]:
    values = dict(os.environ)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT / "workspace_l2_v2")
    parser.add_argument("--questions", type=Path, default=SHARED / "benchmarks" / "l2_film_120_v2" / "test.jsonl")
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / "kg2rag" / "config" / "api.env")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "l2_film_120_v2_global.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    questions = read_jsonl(args.questions)
    questions = questions[: args.limit] if args.limit else questions
    print(json.dumps({
        "method": "Microsoft GraphRAG Global L2 v2", "questions": len(questions),
        "workspace": str(args.workspace), "output": str(args.output),
        "note": "workspace must be indexed from the matching v2 manifest",
    }, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if not args.allow_api:
        raise SystemExit("Pass --allow-api after --dry-run.")
    if not args.workspace.exists():
        raise SystemExit(f"Missing v2 index workspace: {args.workspace}")

    values = envfile(args.env_file)
    process_env = os.environ.copy()
    process_env["GRAPHRAG_LLM_API_KEY"] = values["REPRO_LLM_API_KEY"]
    process_env["GRAPHRAG_EMBED_API_KEY"] = values["REPRO_EMBED_API_KEY"]
    completed = {}
    if args.output.exists():
        completed = {row["id"]: row for row in json.loads(args.output.read_text(encoding="utf-8")).get("results", [])}
    started = time.perf_counter()
    executable = str(ROOT.parent / ".venv-microsoft-graphrag" / "bin" / "graphrag")
    for number, question in enumerate(questions, start=1):
        if question["id"] in completed:
            continue
        query_started = time.perf_counter()
        response = subprocess.run(
            [executable, "query", "--root", str(args.workspace), "--method", "global", question["question"]],
            text=True, capture_output=True, env=process_env,
        )
        answer = response.stdout.strip()
        metrics = score_answer(question, answer)
        completed[question["id"]] = {
            "id": question["id"], "type": question["type"], "question": question["question"],
            "answer": answer, "score": metrics["score"], "metrics": metrics,
            "query_seconds": round(time.perf_counter() - query_started, 3),
            "returncode": response.returncode, "stderr": response.stderr[-1000:],
        }
        rows = [completed[item["id"]] for item in questions if item["id"] in completed]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "method": "Microsoft GraphRAG Global L2 v2", "questions_target": len(questions),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "evaluation": summarize_results(rows), "results": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[query {number:02d}/{len(questions)}] {question['id']}: {metrics['score']:.3f}", flush=True)
    result = json.loads(args.output.read_text(encoding="utf-8"))
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
