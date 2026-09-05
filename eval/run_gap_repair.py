"""Reproducible paired evaluation; model only receives question + visible graph.

--offline builds a separate in-memory hashing index, never overwrites BGE vectors.
--source-dir can point at an isolated, matching frozen corpus checkout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_benchmark_v2 import render_report, source_digest, summarize
from src.core.config import Settings, load_settings
from src.core.types import EdgeMask, RetrievalConstraints
from src.evaluation import BenchmarkScorer, load_benchmark
from src.methods.gap_repair import GapRepairMethod
from src.methods import build_method
from eval.benchmark_methods import BenchmarkHybridMethod
from src.generate.answer import StructuredAnswerGenerator, build_generator
from src.retrieve.dataset_graph import DatasetGraphLoader
from src.retrieve.embedding import HashingEmbedder
from src.retrieve.gap_repair import GapRepairRetriever
from src.retrieve.registry import RetrievalContext
from src.retrieve.vector_index import ChunkVectorIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--method", choices=["gap_repair", "vector", "kg2rag", "hipporag2", "naive_hybrid", "oracle_repair"], default="gap_repair")
    parser.add_argument("--questions", default=str(ROOT / "eval/benchmark_v2/questions.yaml"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--enhanced", action="store_true")
    parser.add_argument("--corpus-records", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--answer-mode", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--graph-view", choices=["both", "complete", "masked"], default="both")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--ablation", choices=["full", "no_repair", "no_compensation", "always", "no_prune"], default="full")
    parser.add_argument("--output", default="eval/results/gap_repair")
    args = parser.parse_args()
    if args.method != "gap_repair" and (args.answer_mode != "llm" or args.offline or args.ablation != "full"):
        parser.error("Baselines require --answer-mode llm, remote embeddings, and --ablation full")
    base_settings = load_settings(args.settings)
    data = copy.deepcopy(base_settings.as_dict())
    if args.source_dir:
        data["paths"]["dataset_dir"] = args.source_dir
    data.setdefault("gap_repair", {})["answer_mode"] = args.answer_mode
    if args.enhanced:
        data["gap_repair"].update({"frontier_queries": True, "max_queries": 32, "search_top_k": 20})
    if args.corpus_records is not None:
        data["gap_repair"]["corpus_records"] = args.corpus_records
    if args.ablation == "no_repair":
        data["gap_repair"]["temporary_edges"] = False
    if args.ablation == "no_compensation":
        data["gap_repair"]["compensation"] = "off"
    if args.ablation == "always":
        data["gap_repair"]["compensation"] = "always"
    if args.ablation == "no_prune":
        data["gap_repair"]["prune"] = False
    settings = Settings(data, base_settings.source)
    benchmark = load_benchmark(Path(args.questions))
    digest = source_digest(settings.path("paths.dataset_dir"))
    if digest != benchmark["benchmark"]["source_sha256"]:
        raise SystemExit("Source digest mismatch: use a matching frozen source directory; do not tune on changed gold.")
    index = ChunkVectorIndex(settings, embedder=HashingEmbedder(1024) if args.offline else None).load()
    if not args.offline and isinstance(index.embedder, HashingEmbedder):
        raise SystemExit("Remote embedding credentials unavailable. Use --offline to rebuild matching hashing vectors in memory.")
    if args.offline:
        index.build(index.chunks)
    store = DatasetGraphLoader(settings, index).load()
    context = RetrievalContext.assemble(store, index, settings)
    generator = build_generator(settings) if args.answer_mode == "llm" else StructuredAnswerGenerator(settings)
    if args.answer_mode == "llm" and not getattr(generator, "ready", False):
        raise SystemExit("Shared LLM generator unavailable; refusing to mislabel a fallback as LLM evaluation.")
    method = GapRepairMethod(settings, retriever=GapRepairRetriever(context), generator=generator)
    if args.method in {"naive_hybrid", "oracle_repair"}:
        method = BenchmarkHybridMethod(settings, oracle=args.method == "oracle_repair")
    elif args.method != "gap_repair":
        method = build_method(args.method, settings)
    scorer = BenchmarkScorer(benchmark)
    metadata = {"source_sha256": digest, "split": args.split, "top_k": args.top_k, "method": args.method,
                "questions_sha256": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
                "embedding": "hashing-1024" if args.offline else settings.get("embedding.model"),
                "generator": args.answer_mode, "ablation": args.ablation,
                "llm_model": settings.get("llm.model") if args.answer_mode == "llm" else None,
                "gap_repair": settings.get("gap_repair"),
                "index_sha256": hashlib.sha256(settings.path("paths.embedding_file").read_bytes()).hexdigest()}
    code_digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        code_digest.update(path.relative_to(ROOT).as_posix().encode())
        code_digest.update(path.read_bytes())
    metadata["code_sha256"] = code_digest.hexdigest()
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / args.output / f"{args.split}-{args.ablation}-{signature}"
    output.mkdir(parents=True, exist_ok=True)
    row_path = output / "results.jsonl"
    rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines()] if row_path.exists() else []
    done = {(r["question_id"], r["graph_view"]) for r in rows if not r.get("error")}
    views = ("complete", "masked") if args.graph_view == "both" else (args.graph_view,)

    def evaluate(view, question):
        try:
            masks = tuple(EdgeMask.from_dict(e) for e in question["graph_perturbation"]["masked_edges"]) if view == "masked" else ()
            oracle_queries = tuple(question["graph_perturbation"].get("oracle_queries", ())) if args.method == "oracle_repair" and view == "masked" else ()
            answer = method.ask(question["question"], top_k=args.top_k, constraints=RetrievalConstraints(masks, oracle_queries))
            return {"question_id": question["id"], "kind": question["kind"], "method": args.method,
                   "graph_view": view, "answer": answer.to_dict(),
                   "metrics": scorer.score(question, answer, view), "error": None}
        except Exception as exc:
            return {"question_id": question["id"], "kind": question["kind"], "method": args.method,
                    "graph_view": view, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = [executor.submit(evaluate, view, question) for view in views for question in benchmark["questions"]
                   if question["split"] == args.split and (question["id"], view) not in done]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            with row_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row["error"]:
                print(row["graph_view"], row["question_id"], "ERROR", row["error"], flush=True)
            else:
                print(row["graph_view"], row["question_id"], "F1", round(row["metrics"]["answer"]["f1"], 3),
                      "recovered", row["metrics"]["retrieval"]["recovered_path_complete_rate"], flush=True)
    rows = list({(r["question_id"], r["graph_view"]): r for r in rows}.values())
    for view in views:
        summary = summarize([r for r in rows if r["graph_view"] == view], metadata)
        (output / f"{view}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / f"{view}.md").write_text(render_report(summary), encoding="utf-8")
    print("Results:", output)
    method.close()
    if any(r.get("error") for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
