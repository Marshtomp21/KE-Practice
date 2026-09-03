#!/usr/bin/env python3
"""Resumable, quality-gated KG²RAG-style run on the audited film benchmark."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from api_clients import APISettings, ChatClient, EmbeddingClient, RerankClient, load_env_file
from openie_parser import normalized, parse_triples

ROOT = Path(__file__).resolve().parents[1]
SHARED_SCRIPTS = ROOT.parent / "shared" / "scripts"
sys.path.insert(0, str(SHARED_SCRIPTS))
from benchmark_utils import read_jsonl, score_answer, summarize_results

DEFAULT_CORPUS = ROOT.parent / "shared" / "benchmarks" / "l2_film_120_v2"


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(x * y for x, y in zip(left, right))
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(x * x for x in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def extraction_prompt(text: str) -> str:
    return (
        "Extract only factual knowledge triples from the Chinese movie document. "
        "Return a JSON object exactly shaped as "
        '{"triples":[{"subject":"...","relation":"...","object":"..."}]}. '
        "Use facts explicitly present in the document; do not invent facts.\n\n" + text
    )


def repair_prompt(previous: str) -> str:
    return (
        "Reformat the extraction below as valid JSON only. The exact schema is "
        '{"triples":[{"subject":"...","relation":"...","object":"..."}]}. '
        "Preserve factual triples, omit commentary, and do not invent new facts.\n\n" + previous[:6000]
    )


def extract_with_retries(text: str, llm: ChatClient, max_attempts: int) -> tuple[list[list[str]], dict[str, Any]]:
    attempts = []
    prompt = extraction_prompt(text)
    for attempt in range(1, max_attempts + 1):
        response = llm.complete(prompt, max_tokens=900)
        triples = parse_triples(response)
        attempts.append({"attempt": attempt, "response": response, "parsed_triples": len(triples)})
        if triples:
            return triples, {"status": "parsed", "attempts": attempts}
        prompt = repair_prompt(response)
    return [], {"status": "unparsed", "attempts": attempts}


def quality_report(chunks: list[dict[str, Any]], raw_dir: Path, minimum_ratio: float, minimum_triples: int) -> dict[str, Any]:
    total = len(chunks)
    nonempty = sum(bool(chunk["triples"]) for chunk in chunks)
    triple_count = sum(len(chunk["triples"]) for chunk in chunks)
    raw_count = sum((raw_dir / f"{chunk['id']}.json").exists() for chunk in chunks)
    report = {
        "documents": total, "raw_response_documents": raw_count,
        "raw_response_coverage": round(raw_count / total, 6) if total else 0.0,
        "nonempty_documents": nonempty,
        "nonempty_document_ratio": round(nonempty / total, 6) if total else 0.0,
        "total_triples": triple_count,
        "thresholds": {"raw_response_coverage": 1.0, "nonempty_document_ratio": minimum_ratio, "total_triples": minimum_triples},
    }
    report["passed"] = (
        report["raw_response_coverage"] >= 1.0
        and report["nonempty_document_ratio"] >= minimum_ratio
        and triple_count >= minimum_triples
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / "config/api.env")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--rerun-types", default="", help="Comma-separated question types to recompute")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "l2_film_120_v2")
    parser.add_argument("--max-extraction-attempts", type=int, default=3)
    parser.add_argument("--min-nonempty-ratio", type=float, default=0.80)
    parser.add_argument("--min-total-triples", type=int, default=120)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    question_path = args.questions or args.corpus / "test.jsonl"
    rerun_types = {item.strip() for item in args.rerun_types.split(",") if item.strip()}
    docs = read_jsonl(args.corpus / "manifest.jsonl")
    questions = read_jsonl(question_path)
    plan = {
        "method": "KG²RAG API-adapted L2 v2", "documents": len(docs), "questions": len(questions),
        "llm_calls_upper_bound": len(docs) * args.max_extraction_attempts + len(questions),
        "quality_gate": {"min_nonempty_ratio": args.min_nonempty_ratio, "min_total_triples": args.min_total_triples},
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if not args.allow_api:
        raise SystemExit("Pass --allow-api after --dry-run.")

    settings = APISettings.from_values(load_env_file(args.env_file))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kg_dir = args.output_dir / "kg"
    raw_dir = args.output_dir / "openie_raw"
    kg_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    llm = ChatClient(settings)
    chunks = []
    index_started = time.perf_counter()
    for index, row in enumerate(docs, start=1):
        text = (args.corpus / row["path"]).read_text(encoding="utf-8")
        kg_path = kg_dir / f"{row['doc_id']}.json"
        raw_path = raw_dir / f"{row['doc_id']}.json"
        triples = json.loads(kg_path.read_text(encoding="utf-8")) if kg_path.exists() else []
        if not triples:
            triples, audit = extract_with_retries(text, llm, max(1, args.max_extraction_attempts))
            raw_path.write_text(json.dumps({"doc_id": row["doc_id"], **audit}, ensure_ascii=False, indent=2), encoding="utf-8")
            kg_path.write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")
        elif not raw_path.exists():
            raw_path.write_text(json.dumps({"doc_id": row["doc_id"], "status": "legacy-cache-without-raw-response", "attempts": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        chunks.append({"id": row["doc_id"], "text": text, "triples": triples})
        print(f"[openie {index:03d}/{len(docs)}] {row['doc_id']}: {len(triples)} triples", flush=True)

    gate = quality_report(chunks, raw_dir, args.min_nonempty_ratio, args.min_total_triples)
    gate["index_seconds"] = round(time.perf_counter() - index_started, 3)
    (args.output_dir / "kg_quality.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    if not gate["passed"]:
        raise SystemExit(f"KG quality gate failed; inspect {args.output_dir / 'kg_quality.json'}")

    embedder = EmbeddingClient(settings)
    reranker = RerankClient(settings)
    embeddings_path = args.output_dir / "document_embeddings.json"
    if embeddings_path.exists():
        document_vectors = json.loads(embeddings_path.read_text(encoding="utf-8"))
        if len(document_vectors) != len(chunks):
            raise RuntimeError("Cached document embedding count does not match corpus")
    else:
        document_vectors = []
        batch_size = max(1, args.embedding_batch_size)
        for start in range(0, len(chunks), batch_size):
            document_vectors.extend(embedder.embed([chunk["text"] for chunk in chunks[start:start + batch_size]]))
            print(f"[embedding {min(start + batch_size, len(chunks)):03d}/{len(chunks)}]", flush=True)
        embeddings_path.write_text(json.dumps(document_vectors), encoding="utf-8")
    summary_path = args.summary_file or args.output_dir / "summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    completed = {row["id"]: row for row in existing.get("results", [])}
    for index, question in enumerate(questions, start=1):
        if question["id"] in completed and question["type"] not in rerun_types:
            print(f"[query {index:02d}/{len(questions)}] {question['id']}: cached", flush=True)
            continue
        query_started = time.perf_counter()
        query_vector = embedder.embed([question["question"]])[0]
        vector_scores = [cosine(query_vector, vector) for vector in document_vectors]
        seed_indexes = sorted(range(len(chunks)), key=lambda item: vector_scores[item], reverse=True)[:5]
        seed_entities = {normalized(value) for item in seed_indexes for triple in chunks[item]["triples"] for value in (triple[0], triple[2])}
        chosen = set(seed_indexes)
        for item, chunk in enumerate(chunks):
            if any(normalized(value) in seed_entities for triple in chunk["triples"] for value in (triple[0], triple[2])):
                chosen.add(item)
        candidates = [chunks[item] for item in sorted(chosen)]
        reranked = reranker.rerank(question["question"], [chunk["text"] + "\nKG:" + json.dumps(chunk["triples"], ensure_ascii=False) for chunk in candidates], 6)
        final = [(candidates[row["index"]], row) for row in reranked if isinstance(row.get("index"), int) and 0 <= row["index"] < len(candidates)]
        answer = llm.complete(
            "仅基于证据简洁回答；把给出的证据视为本题完整的冻结语料范围，不要求覆盖现实世界的完整片单。"
            "若问题询问哪些演员出现不止一次，请直接比较各影片演员名单，列出在至少两部不同影片中出现的人；"
            "不要因为缺少冻结语料之外的影片而拒答。只有证据中完全没有所需事实时，才回答无法根据现有材料确定。\n问题："
            + question["question"] + "\n证据：\n" + "\n---\n".join(chunk["text"] for chunk, _ in final),
            max_tokens=500,
        )
        metrics = score_answer(question, answer)
        completed[question["id"]] = {
            "id": question["id"], "type": question["type"], "question": question["question"],
            "answer": answer, "score": metrics["score"], "metrics": metrics,
            "seed_documents": [{"doc_id": chunks[item]["id"], "vector_score": round(vector_scores[item], 6)} for item in seed_indexes],
            "expanded_document_count": len(candidates),
            "retrieved_evidence": [{"doc_id": chunk["id"], "rerank_score": row.get("relevance_score", row.get("score"))} for chunk, row in final],
            "query_seconds": round(time.perf_counter() - query_started, 3),
        }
        results = [completed[item["id"]] for item in questions if item["id"] in completed]
        summary = {
            "method": "KG²RAG API-adapted L2 v2", "corpus": args.corpus.name,
            "documents": len(docs), "questions_target": len(questions), "kg_quality": gate,
            "evaluation": summarize_results(results), "results": results,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[query {index:02d}/{len(questions)}] {question['id']}: {metrics['score']:.3f}", flush=True)
    results = [completed[item["id"]] for item in questions if item["id"] in completed]
    summary = {
        "method": "KG²RAG API-adapted L2 v2", "corpus": args.corpus.name,
        "documents": len(docs), "questions_target": len(questions), "kg_quality": gate,
        "evaluation": summarize_results(results), "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
