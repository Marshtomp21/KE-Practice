#!/usr/bin/env python3
"""Run HippoRAG 2 with separate LLM and embedding API credentials."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from hipporag import HippoRAG
from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.utils.config_utils import BaseConfig

SHARED_SCRIPTS = ROOT.parent / "shared" / "scripts"
sys.path.insert(0, str(SHARED_SCRIPTS))
from benchmark_utils import json_safe, read_jsonl, score_answer, summarize_results

DEFAULT_CORPUS = ROOT.parent / "shared" / "benchmarks" / "l2_film_120_v2"


def load_env(path: Path) -> Dict[str, str]:
    values = dict(os.environ)
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@contextmanager
def temporary_openai_key(key: str) -> Iterator[None]:
    old: Optional[str] = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = key
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old


def required(values: Dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise SystemExit(f"Missing {key}; fill config/api.env locally.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / "config/api.env")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "l2_film_120_v2")
    parser.add_argument("--corpus", type=Path, default=None, help="Audited benchmark directory; omit for L1.")
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--l2", action="store_true")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.l2 and args.corpus is None:
        args.corpus = DEFAULT_CORPUS
    manifest = read_jsonl(args.corpus / "manifest.jsonl") if args.corpus else None
    question_path = args.questions or (args.corpus / "test.jsonl" if args.corpus else None)
    questions = read_jsonl(question_path) if question_path else None
    plan = {
        "method": "HippoRAG 2 official workflow with injected dual-provider clients",
        "documents": len(manifest) if manifest else 1, "questions": len(questions) if questions else 1,
        "force_reindex": args.force_reindex, "openie_max_workers": 1,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if not args.allow_api:
        raise SystemExit("Refusing remote calls; inspect --dry-run then pass --allow-api.")

    values = load_env(args.env_file)
    llm_key = required(values, "HIPPO_LLM_API_KEY")
    embed_key = required(values, "HIPPO_EMBED_API_KEY")
    config = BaseConfig(
        save_dir=str(args.output_dir), llm_name=values.get("HIPPO_LLM_MODEL", "deepseek-v4-flash"),
        llm_base_url=values.get("HIPPO_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        embedding_model_name=values.get("HIPPO_EMBED_MODEL", "BAAI/bge-m3"),
        embedding_base_url=values.get("HIPPO_EMBED_BASE_URL", "https://api.siliconflow.cn/v1"),
        embedding_provider="openai", openie_max_workers=1, max_retry_attempts=1,
        force_index_from_scratch=args.force_reindex,
    )
    with temporary_openai_key(llm_key):
        llm = CacheOpenAI.from_experiment_config(config)
    llm.llm_config.generate_params["extra_body"] = {"thinking": {"type": "disabled"}}
    with temporary_openai_key(embed_key):
        embedder = OpenAIEmbeddingModel(config, config.embedding_model_name)

    try:
        with HippoRAG(
            global_config=config, extraction_llm=llm, qa_llm=llm, embedding_model=embedder,
            index_identity="deepseek-v4-flash__siliconflow-bge-m3__v2",
        ) as rag:
            docs = [(args.corpus / row["path"]).read_text(encoding="utf-8") for row in manifest] if manifest else ["George Rankin is a politician."]
            doc_ids = {text: row["doc_id"] for text, row in zip(docs, manifest or [])}
            index_started = time.perf_counter()
            rag.index(docs=docs)
            index_seconds = round(time.perf_counter() - index_started, 3)
            if not args.l2 or not questions:
                solutions, messages, metadata = rag.rag_qa(queries=["What is George Rankin's occupation?"])
                print(json.dumps({"solution": json_safe(solutions[0]), "message": json_safe(messages[0]), "metadata": json_safe(metadata[0]), "index_seconds": index_seconds}, ensure_ascii=False, indent=2))
                return

            args.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = args.output_dir / "summary.json"
            existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            completed = {row["id"]: row for row in existing.get("results", [])}
            summary = {
                "method": "HippoRAG 2 API-adapted L2 v2", "corpus": args.corpus.name,
                "documents": len(docs), "questions_target": len(questions), "index_call_seconds": index_seconds,
                "evaluation": summarize_results(list(completed.values())),
                "results": [completed[item["id"]] for item in questions if item["id"] in completed],
            }
            for number, question in enumerate(questions, start=1):
                if question["id"] in completed:
                    print(f"[query {number:02d}/{len(questions)}] {question['id']}: cached", flush=True)
                    continue
                query_started = time.perf_counter()
                # Official contract: (query_solutions, response_messages, metadata).
                query_solutions, response_messages, all_metadata = rag.rag_qa(queries=[question["question"]])
                if len(query_solutions) != 1:
                    raise RuntimeError(f"Expected one QuerySolution, received {len(query_solutions)}")
                solution = query_solutions[0]
                answer = str(solution.answer or (response_messages[0] if response_messages else ""))
                scores = json_safe(solution.doc_scores) or []
                retrieved = []
                for index, text in enumerate(solution.docs[:5]):
                    retrieved.append({
                        "rank": index + 1, "doc_id": doc_ids.get(text),
                        "score": scores[index] if index < len(scores) else None,
                        "snippet": str(text)[:500],
                    })
                metrics = score_answer(question, answer)
                completed[question["id"]] = {
                    "id": question["id"], "type": question["type"], "question": question["question"],
                    "answer": answer, "score": metrics["score"], "metrics": metrics,
                    "retrieved_evidence": retrieved, "graph_seeds": json_safe(solution.graph_seeds),
                    "response_message": json_safe(response_messages[0] if response_messages else None),
                    "metadata": json_safe(all_metadata[0] if all_metadata else None),
                    "query_seconds": round(time.perf_counter() - query_started, 3),
                }
                rows = [completed[item["id"]] for item in questions if item["id"] in completed]
                summary = {
                    "method": "HippoRAG 2 API-adapted L2 v2", "corpus": args.corpus.name,
                    "documents": len(docs), "questions_target": len(questions), "index_call_seconds": index_seconds,
                    "evaluation": summarize_results(rows), "results": rows,
                }
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[query {number:02d}/{len(questions)}] {question['id']}: {metrics['score']:.3f}", flush=True)
            # Refresh the summary even when all questions came from a checkpoint.
            rows = [completed[item["id"]] for item in questions if item["id"] in completed]
            summary["evaluation"] = summarize_results(rows)
            summary["results"] = rows
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
    finally:
        llm.close()
        embedder.close()


if __name__ == "__main__":
    main()
