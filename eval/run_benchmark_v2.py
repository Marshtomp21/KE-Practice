"""运行冻结的 40 题 benchmark，并逐题保存可恢复、可审计结果。"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import BenchmarkScorer, load_benchmark
from src.core.config import load_settings
from src.core.types import EdgeMask, RetrievalConstraints
from src.generate.service import QAService
from eval.benchmark_methods import BenchmarkHybridMethod

DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "benchmark_v2" / "questions.yaml"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "eval" / "results" / "benchmark_v2"
SOURCE_DIR = PROJECT_ROOT / "data" / "source" / "wikipedia_300_films_final"


def source_digest() -> str:
    digest = hashlib.sha256()
    for name in ("films.jsonl", "actors.jsonl", "directors.jsonl", "relations.jsonl"):
        digest.update(name.encode())
        digest.update((SOURCE_DIR / name).read_bytes())
    return digest.hexdigest()


def mean(rows: Iterable[dict], path: str) -> float | None:
    keys = path.split(".")
    values = []
    for row in rows:
        value: Any = row
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.mean(values) if values else None


def summarize(rows: List[dict], metadata: dict) -> dict:
    result: Dict[str, Any] = {"run": metadata, "methods": {}}
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            by_method[row["method"]].append(row)
    fields = {
        "answer_f1": "metrics.answer.f1",
        "exact_match": "metrics.answer.exact_match",
        "document_recall": "metrics.retrieval.document_recall",
        "document_f1": "metrics.retrieval.document_f1",
        "relation_recall": "metrics.retrieval.relation_recall",
        "path_complete_rate": "metrics.retrieval.path_complete_rate",
        "recovered_path_complete_rate": "metrics.retrieval.recovered_path_complete_rate",
        "gap_detection_accuracy": "metrics.gap.gap_detection_correct",
        "compensation_document_recall": "metrics.gap.compensation_document_recall",
        "unnecessary_compensation": "metrics.gap.unnecessary_compensation",
        "legacy_gold_substring_recall": "metrics.diagnostics.legacy_gold_substring_recall",
        "latency": "answer.latency",
    }
    for method, subset in by_method.items():
        kinds: Dict[str, List[dict]] = defaultdict(list)
        for row in subset:
            kinds[row["kind"]].append(row)
        result["methods"][method] = {
            "completed": len(subset),
            "overall": {name: mean(subset, field) for name, field in fields.items()},
            "by_kind": {
                kind: {
                    "count": len(items),
                    "answer_f1": mean(items, "metrics.answer.f1"),
                    "exact_match": mean(items, "metrics.answer.exact_match"),
                    "document_recall": mean(items, "metrics.retrieval.document_recall"),
                    "path_complete_rate": mean(items, "metrics.retrieval.path_complete_rate"),
                }
                for kind, items in sorted(kinds.items())
            },
        }
    return result


def render_report(summary: dict) -> str:
    lines = ["# Benchmark v2 实验结果", "", "## 总体结果", "",
             "| 方法 | 完成题数 | Answer F1 | EM | Gold Doc R | 可见路径完整率 | 修复后路径完整率 | 缺口检测准确率 | 补偿 Doc R | 平均耗时(s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, item in summary.get("methods", {}).items():
        overall = item["overall"]
        fmt = lambda key: "-" if overall.get(key) is None else f"{overall[key]:.3f}"
        lines.append(
            f"| {method} | {item['completed']} | {fmt('answer_f1')} | {fmt('exact_match')} | "
            f"{fmt('document_recall')} | {fmt('path_complete_rate')} | "
            f"{fmt('recovered_path_complete_rate')} | {fmt('gap_detection_accuracy')} | "
            f"{fmt('compensation_document_recall')} | {fmt('latency')} |"
        )
    lines.extend(["", "## 分题型 Answer F1", ""])
    methods = list(summary.get("methods", {}))
    kinds = sorted({kind for item in summary.get("methods", {}).values() for kind in item["by_kind"]})
    lines.append("| 题型 | " + " | ".join(methods) + " |")
    lines.append("|---|" + "---:|" * len(methods))
    for kind in kinds:
        values = []
        for method in methods:
            value = summary["methods"][method]["by_kind"].get(kind, {}).get("answer_f1")
            values.append("-" if value is None else f"{value:.3f}")
        lines.append(f"| {kind} | " + " | ".join(values) + " |")
    lines.extend([
        "", "## 指标说明", "",
        "- `Answer F1`：别名归一化后的实体集合 F1；多答会降低 Precision。",
        "- `Exact Match`：预测实体集合与 gold 完全相等；无答案题要求明确否定，材料不足不计正确。",
        "- `Gold Doc Recall`：返回引用覆盖最小 gold 证据文档的比例。",
        "- `Path Complete`：返回子图完整覆盖必要关系路径的比例；纯向量方法没有子图，因此该项为 0。",
        "- `修复后路径完整率`：把仅用于本次查询的临时证据关系计入后，gold 路径被恢复的比例。",
        "- `缺口检测准确率`：是否正确判断当前查询视图存在需要补偿的关系缺口。",
        "- `补偿 Doc Recall`：补偿检索返回的文档覆盖缺边 gold 文档的比例。",
        "- `旧子串召回`：模拟旧规则仅看 gold 名称是否出现在全文，用于诊断旧 scorer 是否虚高。",
    ])
    return "\n".join(lines) + "\n"


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument(
        "--methods",
        default="vector,library_graphrag,kg2rag,hipporag2,naive_hybrid,oracle_repair",
    )
    parser.add_argument(
        "--graph-view", choices=("masked", "complete"), default="masked",
        help="masked 按题目隐藏指定关系；complete 使用完整图作为配对对照",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--fresh", action="store_true", help="忽略已有 JSONL，从头运行")
    parser.add_argument("--allow-source-drift", action="store_true")
    args = parser.parse_args()

    payload = load_benchmark(args.questions)
    expected_digest = str(payload["benchmark"].get("source_sha256") or "")
    actual_digest = source_digest()
    if expected_digest != actual_digest and not args.allow_source_drift:
        raise SystemExit(
            "源语料与冻结 benchmark 的 source_sha256 不一致；请先检查变更并重新标注，"
            "或显式传入 --allow-source-drift。"
        )
    questions = list(payload["questions"])
    if args.split != "all":
        questions = [item for item in questions if item.get("split") == args.split]
    if args.limit:
        questions = questions[: args.limit]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    supported = {"vector", "library_graphrag", "kg2rag", "hipporag2",
                 "naive_hybrid", "oracle_repair"}
    unknown = sorted(set(methods) - supported)
    if unknown:
        raise SystemExit(f"未知 benchmark 方法：{unknown}；可用方法：{sorted(supported)}")
    settings = load_settings()
    run_config = {
        "llm_endpoint": settings.get("llm.endpoint"),
        "llm_model": settings.get("llm.model"),
        "embedding_endpoint": settings.get("embedding.endpoint"),
        "embedding_model": settings.get("embedding.model"),
        "generation_temperature": settings.get("generation.temperature"),
        "top_k": args.top_k,
    }
    run_signature = hashlib.sha256(
        json.dumps(run_config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result_dir = Path(args.result_dir) / args.graph_view
    result_dir.mkdir(parents=True, exist_ok=True)
    row_file = result_dir / "results.jsonl"
    if args.fresh and row_file.exists():
        row_file.unlink()
    stored_rows = read_rows(row_file)
    selected_ids = {item["id"] for item in questions}
    rows = [
        row for row in stored_rows
        if row.get("question_id") in selected_ids
        and row.get("method") in methods
        and row.get("top_k") == args.top_k
        and row.get("source_sha256") == expected_digest
        and row.get("graph_view") == args.graph_view
        and row.get("run_signature") == run_signature
    ]
    completed = {
        (row["question_id"], row["method"], row["graph_view"])
        for row in rows if not row.get("error")
    }
    scorer = BenchmarkScorer(payload)
    service = QAService(settings)
    benchmark_methods: Dict[str, BenchmarkHybridMethod] = {}
    metadata = {
        "benchmark": payload["benchmark"],
        "methods": methods,
        "top_k": args.top_k,
        "split": args.split,
        "graph_view": args.graph_view,
        "configuration": run_config,
        "run_signature": run_signature,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"benchmark {len(questions)} 题 × {len(methods)} 方法；已有 {len(completed)} 条可复用结果")
    try:
        for method in methods:
            for question in questions:
                key = (question["id"], method, args.graph_view)
                if key in completed:
                    continue
                print(f"[{method}] {question['id']} ...", flush=True)
                try:
                    perturbation = question.get("graph_perturbation") or {}
                    masked_edges = (
                        tuple(EdgeMask.from_dict(edge) for edge in perturbation.get("masked_edges") or [])
                        if args.graph_view == "masked" else ()
                    )
                    supplemental_queries = (
                        tuple(str(item) for item in perturbation.get("oracle_queries") or [])
                        if args.graph_view == "masked" else ()
                    )
                    constraints = RetrievalConstraints(masked_edges, supplemental_queries)
                    if method in {"naive_hybrid", "oracle_repair"}:
                        if method not in benchmark_methods:
                            benchmark_methods[method] = BenchmarkHybridMethod(
                                settings, oracle=method == "oracle_repair"
                            )
                        answer = benchmark_methods[method].ask(
                            question["question"], top_k=args.top_k, constraints=constraints
                        )
                    else:
                        answer = service.ask(
                            question["question"], retriever_name=method,
                            top_k=args.top_k, constraints=constraints,
                        )
                    row = {
                        "question_id": question["id"], "kind": question["kind"],
                        "split": question.get("split", "test"),
                        "top_k": args.top_k, "source_sha256": expected_digest,
                        "run_signature": run_signature,
                        "question": question["question"], "method": method,
                        "graph_view": args.graph_view,
                        "answer": answer.to_dict(),
                        "metrics": scorer.score(question, answer, graph_view=args.graph_view),
                        "error": None,
                    }
                    print(
                        f"  F1={row['metrics']['answer']['f1']:.3f} "
                        f"EM={row['metrics']['answer']['exact_match']:.0f} "
                        f"DocR={row['metrics']['retrieval']['document_recall']:.3f} "
                        f"{answer.latency:.2f}s",
                        flush=True,
                    )
                except Exception as exc:
                    row = {
                        "question_id": question["id"], "kind": question["kind"],
                        "split": question.get("split", "test"),
                        "top_k": args.top_k, "source_sha256": expected_digest,
                        "run_signature": run_signature,
                        "question": question["question"], "method": method,
                        "graph_view": args.graph_view,
                        "answer": None, "metrics": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"  ERROR {row['error']}", flush=True)
                with row_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
    finally:
        service.close()
        for method in benchmark_methods.values():
            method.close()

    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    # 同一项可能先失败、续跑后成功；报告只采用最后一次状态，不重复计数历史记录。
    latest = {
        (row["question_id"], row["method"], row.get("graph_view")): row for row in rows
    }
    final_rows = list(latest.values())
    summary = summarize(final_rows, metadata)
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (result_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    errors = [row for row in final_rows if row.get("error")]
    print(f"结果 -> {row_file}\n汇总 -> {result_dir / 'report.md'}\n错误 {len(errors)} 条")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
