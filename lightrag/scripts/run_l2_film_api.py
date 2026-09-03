#!/usr/bin/env python3
"""Run LightRAG on the frozen shared L2 film corpus.

All document and question inputs come from ``reproductions/shared``.  This is
an API-adapted run because completion and embedding models are hosted remotely,
while LightRAG indexing/retrieval code remains the official installed package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT.parent / "kg2rag" / "config" / "api.env"
DEFAULT_CORPUS = ROOT.parent / "shared" / "benchmarks" / "l2_film_120_v2"
sys.path.insert(0, str(ROOT.parent / "shared" / "scripts"))
from benchmark_utils import read_jsonl, score_answer, summarize_results


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def required(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise SystemExit(f"Missing {key} in {DEFAULT_ENV}.")
    return value


async def run(values: dict[str, str], corpus: Path, questions_path: Path, output_dir: Path, mode: str) -> dict[str, Any]:
    manifest = read_jsonl(corpus / "manifest.jsonl")
    questions = read_jsonl(questions_path)
    documents = [(corpus / row["path"]).read_text(encoding="utf-8") for row in manifest]

    async def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        return await openai_complete_if_cache(
            values.get("REPRO_LLM_MODEL", "deepseek-v4-flash"), prompt,
            system_prompt=system_prompt, history_messages=history_messages or [],
            api_key=required(values, "REPRO_LLM_API_KEY"),
            base_url=values.get("REPRO_LLM_ENDPOINT", "https://api.deepseek.com/chat/completions").rsplit("/chat/completions", 1)[0],
            **kwargs,
        )

    embedding = EmbeddingFunc(
        embedding_dim=1024, max_token_size=8192,
        model_name=values.get("REPRO_EMBED_MODEL", "BAAI/bge-m3"),
        func=partial(openai_embed.func, model=values.get("REPRO_EMBED_MODEL", "BAAI/bge-m3"),
                     base_url=values.get("REPRO_EMBED_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings").rsplit("/embeddings", 1)[0],
                     api_key=required(values, "REPRO_EMBED_API_KEY")),
    )
    rag = LightRAG(working_dir=str(output_dir / "storage"), llm_model_func=llm, embedding_func=embedding)
    await rag.initialize_storages()
    started = time.perf_counter()
    try:
        await rag.ainsert(documents, ids=[row["doc_id"] for row in manifest])
        index_seconds = round(time.perf_counter() - started, 3)
        results = []
        for item in questions:
            query_started = time.perf_counter()
            answer = await rag.aquery(item["question"], param=QueryParam(mode=mode, enable_rerank=False, top_k=8, chunk_top_k=8))
            text = str(answer)
            metrics = score_answer(item, text)
            results.append({"id": item["id"], "type": item["type"], "question": item["question"], "reference_answer": item["reference_answer"], "answer": text, "score": metrics["score"], "metrics": metrics, "latency_seconds": round(time.perf_counter() - query_started, 3)})
    finally:
        await rag.finalize_storages()
    summary = {"method": "LightRAG v1.5.7 API-adapted L2 v2", "corpus": corpus.name, "documents": len(manifest), "questions": len(questions), "mode": mode, "index_seconds": index_seconds, "evaluation": summarize_results(results), "mean_query_seconds": round(sum(row["latency_seconds"] for row in results) / len(results), 3), "models": {"llm": values.get("REPRO_LLM_MODEL"), "embedding": values.get("REPRO_EMBED_MODEL")}, "results": results}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "l2_film_120_v2")
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid", "mix"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    manifest = read_jsonl(args.corpus / "manifest.jsonl")
    questions_path = args.questions or args.corpus / "test.jsonl"
    questions = read_jsonl(questions_path)
    print(json.dumps({"method": "LightRAG L2", "documents": len(manifest), "questions": len(questions), "mode": args.mode, "output_dir": str(args.output_dir), "remote_calls": "indexing plus 20 queries"}, ensure_ascii=False))
    if args.dry_run:
        return
    if not args.allow_api:
        raise SystemExit("Refusing remote calls; inspect --dry-run then pass --allow-api.")
    if not args.source_env.exists():
        raise SystemExit(f"Credential source not found: {args.source_env}")
    summary = asyncio.run(run(load_env(args.source_env), args.corpus, questions_path, args.output_dir, args.mode))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
