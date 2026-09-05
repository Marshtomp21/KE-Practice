"""命令行问答。

用法：
  python scripts/ask.py "某位导演执导过哪些影片"
  python scripts/ask.py "A 和 B 有过合作吗" --retriever library_graphrag --show-debug
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate.service import QAService
from src.methods.library_graphrag import MethodUnavailable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="要问的问题")
    parser.add_argument(
        "--retriever", default=None,
        help="问答方法：vector / kg2rag / hipporag2 / gap_repair / library_graphrag（默认读取 settings.yaml）",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--show-debug", action="store_true", help="打印检索过程信息")
    args = parser.parse_args()

    service = QAService()
    print(f"可用检索器：{', '.join(service.retriever_names)}")
    try:
        answer = service.ask(args.question, retriever_name=args.retriever, top_k=args.top_k)
    except MethodUnavailable as exc:
        print(f"方法不可用：{exc}")
        service.close()
        return 2

    print(f"\n检索器：{answer.retriever_name}    耗时：{answer.latency:.2f}s")
    print("-" * 60)
    print(answer.text)
    print("-" * 60)

    if answer.citations:
        print("引用来源：")
        for citation in answer.citations:
            print(
                f"  [{citation.marker}] {citation.doc_id} "
                f"[{citation.char_start}:{citation.char_end}] {citation.snippet[:60]}"
            )

    subgraph = answer.subgraph
    print(f"\n子图：节点 {len(subgraph.entities)}，边 {len(subgraph.relations)}")
    for entity_id in subgraph.highlight_path[:5]:
        entity = next((e for e in subgraph.entities if e.id == entity_id), None)
        if entity:
            print(f"  高分节点 {entity.name}（{entity.type}） {subgraph.node_scores.get(entity_id, 0):.3f}")

    if args.show_debug:
        print("\n调试信息：")
        print(json.dumps(answer.debug_info, ensure_ascii=False, indent=2)[:3000])
    service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
