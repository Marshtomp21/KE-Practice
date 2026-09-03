"""对比实验：同一问题集 × 全部检索器，一条命令跑完并落盘。

公平性由构造保证：所有检索器共享同一个 QAService（同一份语料、同一份切分、
同一个 embedder、同一个生成器、同一个 top_k），唯一的变量就是检索器本身。

产出三个文件：
  eval/results/compare_table.md   按题型分组的对比表，可直接贴进报告
  eval/results/compare_rows.csv   逐题逐检索器的明细
  eval/results/evidence.md        每题各检索器实际检索到的证据内容
用法：python eval/run_compare.py [--retrievers vector,library_graphrag] [--top-k 6]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Answer
from src.generate.service import QAService

QUESTION_FILE = PROJECT_ROOT / "eval" / "questions.yaml"
RESULT_DIR = PROJECT_ROOT / "eval" / "results"

DENIAL_MARKERS = ("没有", "不存在", "查不到", "未", "无", "否")
ASSERTION_MARKERS = ("出演", "参演", "主演", "存在")
# 是非题只看"结论"句：答案正文里的关系清单与原文引用都可能同时出现两个对象，
# 那是罗列证据，不是在断言两者有关系；把它们计入判分会把正确引用误判成幻觉。
VERDICT_PREFIX = "结论"


@dataclass
class Row:
    question_id: str
    kind: str
    question: str
    retriever: str
    score: float
    hits: List[str] = field(default_factory=list)
    latency: float = 0.0
    chunk_count: int = 0
    relation_count: int = 0
    answer: str = ""


def grade(question: Dict[str, Any], answer: Answer) -> tuple[float, List[str]]:
    """按题目声明的判分方式打分，返回 (得分, 命中的期望条目)。"""
    text = answer.text
    expect: Sequence[str] = question.get("expect", [])
    mode = question.get("expect_mode", "contains_any")

    if mode == "denial":
        # 只在"结论"句上判分：关系清单里同时出现两个对象是罗列，不是断言。
        # 拿不出结论句的答案记 0 分——它既没有否定，也没有正面回答，
        # 谈不上抑制幻觉。这条口径对所有检索器一视同仁。
        verdicts = [
            line
            for line in text.splitlines()
            if line.strip().startswith(VERDICT_PREFIX) or VERDICT_PREFIX in line[:8]
        ]
        if not verdicts:
            return 0.0, []
        body = "\n".join(verdicts)
        denied = any(marker in body for marker in DENIAL_MARKERS)
        subjects = question.get("subjects", [])
        asserted = (
            len(subjects) == 2
            and any(
                all(subject in line for subject in subjects)
                and any(marker in line for marker in ASSERTION_MARKERS)
                and not any(marker in line for marker in DENIAL_MARKERS)
                for line in verdicts
            )
        )
        return (1.0 if denied and not asserted else 0.0), (["否定成立"] if denied else [])

    if not expect:
        return 0.0, []
    hits = [item for item in expect if item in text]
    return len(hits) / len(expect), hits


def render_table(rows: Sequence[Row], retrievers: Sequence[str]) -> str:
    by_kind: Dict[str, Dict[str, List[Row]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_kind[row.kind][row.retriever].append(row)

    lines: List[str] = ["# 检索方式对比结果", ""]
    lines.append("得分口径：`contains_any` 题为期望条目命中率，`denial` 题答对记 1 分。")
    lines.append("")
    lines.append("## 按题型分组")
    lines.append("")
    header = "| 题型 | 题数 | " + " | ".join(retrievers) + " |"
    lines.append(header)
    lines.append("|---|---|" + "---|" * len(retrievers))
    for kind in sorted(by_kind):
        counts = len(by_kind[kind][retrievers[0]]) if retrievers else 0
        cells = []
        for name in retrievers:
            scores = [r.score for r in by_kind[kind][name]]
            cells.append(f"{statistics.mean(scores):.2f}" if scores else "-")
        lines.append(f"| {kind} | {counts} | " + " | ".join(cells) + " |")

    overall = []
    for name in retrievers:
        scores = [r.score for r in rows if r.retriever == name]
        overall.append(f"{statistics.mean(scores):.2f}" if scores else "-")
    lines.append("| **总计** | " + str(len({r.question_id for r in rows})) + " | " + " | ".join(overall) + " |")

    lines.append("")
    lines.append("## 平均检索规模与耗时")
    lines.append("")
    lines.append("| 检索器 | 平均片段数 | 平均关系数 | 平均耗时(s) |")
    lines.append("|---|---|---|---|")
    for name in retrievers:
        subset = [r for r in rows if r.retriever == name]
        if not subset:
            continue
        lines.append(
            f"| {name} | {statistics.mean(r.chunk_count for r in subset):.1f} | "
            f"{statistics.mean(r.relation_count for r in subset):.1f} | "
            f"{statistics.mean(r.latency for r in subset):.2f} |"
        )

    lines.append("")
    lines.append("## 逐题得分")
    lines.append("")
    lines.append("| 题号 | 题型 | 问题 | " + " | ".join(retrievers) + " |")
    lines.append("|---|---|---|" + "---|" * len(retrievers))
    for question_id in sorted({r.question_id for r in rows}):
        subset = {r.retriever: r for r in rows if r.question_id == question_id}
        sample = next(iter(subset.values()))
        cells = [f"{subset[name].score:.2f}" if name in subset else "-" for name in retrievers]
        lines.append(
            f"| {question_id} | {sample.kind} | {sample.question} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"


def render_evidence(rows: Sequence[Row], answers: Dict[tuple, Answer]) -> str:
    lines: List[str] = ["# 各检索器实际检索到的证据", ""]
    for question_id in sorted({r.question_id for r in rows}):
        subset = [r for r in rows if r.question_id == question_id]
        sample = subset[0]
        lines.append(f"## {question_id} · {sample.kind}")
        lines.append("")
        lines.append(f"**问题**：{sample.question}")
        lines.append("")
        for row in subset:
            answer = answers[(row.question_id, row.retriever)]
            lines.append(f"### 检索器 `{row.retriever}` — 得分 {row.score:.2f}")
            lines.append("")
            lines.append(f"答案：\n\n> {answer.text.strip().replace(chr(10), chr(10) + '> ')}")
            lines.append("")
            if answer.subgraph.relations:
                names = {e.id: e.name for e in answer.subgraph.entities}
                shown = answer.subgraph.relations[:8]
                lines.append("命中关系：")
                for relation in shown:
                    lines.append(
                        f"- {names.get(relation.head_id, relation.head_id)} "
                        f"--{relation.type}--> {names.get(relation.tail_id, relation.tail_id)}"
                    )
                lines.append("")
            if answer.citations:
                lines.append("命中片段：")
                for citation in answer.citations[:4]:
                    lines.append(
                        f"- `[{citation.marker}]` {citation.doc_id} "
                        f"[{citation.char_start}:{citation.char_end}]：{citation.snippet[:100]}"
                    )
                lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrievers", default="vector",
        help="逗号分隔；库方法已配置时可传 vector,library_graphrag",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--questions", default=str(QUESTION_FILE))
    args = parser.parse_args()

    question_file = Path(args.questions)
    if not question_file.exists():
        print(f"缺少问题集 {question_file}，请先运行 scripts/make_questions.py")
        return 1
    payload = yaml.safe_load(question_file.read_text(encoding="utf-8")) or {}
    questions = payload.get("questions", [])
    if not questions:
        print("问题集为空")
        return 1

    service = QAService()
    retrievers = (
        [name.strip() for name in args.retrievers.split(",") if name.strip()]
        or service.retriever_names
    )
    print(f"问题 {len(questions)} 题 × 检索器 {len(retrievers)} 种 = {len(questions) * len(retrievers)} 次问答")

    rows: List[Row] = []
    answers: Dict[tuple, Answer] = {}
    for question in questions:
        for name in retrievers:
            answer = service.ask(question["question"], retriever_name=name, top_k=args.top_k)
            score, hits = grade(question, answer)
            answers[(question["id"], name)] = answer
            rows.append(
                Row(
                    question_id=question["id"],
                    kind=question["kind"],
                    question=question["question"],
                    retriever=name,
                    score=score,
                    hits=hits,
                    latency=answer.latency,
                    chunk_count=len(answer.citations),
                    relation_count=len(answer.subgraph.relations),
                    answer=answer.text,
                )
            )
            print(f"  {question['id']:<16} {name:<10} 得分 {score:.2f}  耗时 {answer.latency:.2f}s")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table_file = RESULT_DIR / "compare_table.md"
    table_file.write_text(render_table(rows, retrievers), encoding="utf-8")

    csv_file = RESULT_DIR / "compare_rows.csv"
    with csv_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["question_id", "kind", "question", "retriever", "score", "hits", "latency", "chunks", "relations"]
        )
        for row in rows:
            writer.writerow(
                [row.question_id, row.kind, row.question, row.retriever, f"{row.score:.4f}",
                 "|".join(row.hits), f"{row.latency:.3f}", row.chunk_count, row.relation_count]
            )

    evidence_file = RESULT_DIR / "evidence.md"
    evidence_file.write_text(render_evidence(rows, answers), encoding="utf-8")

    print("\n" + render_table(rows, retrievers).split("## 平均检索规模")[0])
    print(f"对比表 -> {table_file}\n明细   -> {csv_file}\n证据   -> {evidence_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
