"""构建 vector、KG²RAG 和 HippoRAG 2 共用的 Chunk 向量索引。

用法：
  python scripts/build_index.py --skip-ingest  # 复用已落盘的切分结果

KG²RAG/HippoRAG 2 的本地图在首次问答时从结构化电影数据只读构建；
neo4j-graphrag 使用 Neo4j 内已有的图与向量索引，不由本脚本构建。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_settings
from src.ingest.pipeline import IngestPipeline, iter_persisted_documents, load_persisted_chunks
from src.retrieve.vector_index import ChunkVectorIndex


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true", help="复用已有的切分结果")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个片段，用于快速试跑")
    args = parser.parse_args()

    settings = load_settings()

    section("第 1 步：导入、清洗与切分")
    if args.skip_ingest:
        documents = list(iter_persisted_documents(settings))
        chunks = load_persisted_chunks(settings)
        print(f"复用已落盘结果：文档 {len(documents)}，片段 {len(chunks)}")
    else:
        pipeline = IngestPipeline(settings)
        documents, chunks, load_report = pipeline.run()
        print(load_report.summary())
        for item in load_report.skipped[:5]:
            print(f"  跳过 {item['source']}: {item['reason']}")
        written = pipeline.persist(documents, chunks)
        print(f"文档 {len(documents)}，片段 {len(chunks)} -> {written['chunks']}")

    if not chunks:
        print("没有可处理的片段，请先运行 scripts/make_sample_corpus.py 准备语料")
        return 1
    if args.limit:
        chunks = chunks[: args.limit]
        kept = {c.doc_id for c in chunks}
        documents = [d for d in documents if d.doc_id in kept]
        print(f"已按 --limit 截断为 {len(chunks)} 个片段")

    section("第 2 步：构建片段向量索引")
    index = ChunkVectorIndex(settings=settings)
    index.build(chunks)
    print(f"向量索引：{len(chunks)} 段，维度 {index.dimension} -> {index.persist()}")

    section("完成")
    print("接下来可以运行：")
    print("  python scripts/ask.py \"某位导演执导过哪些影片\"")
    print("  python eval/run_compare.py --retrievers vector,kg2rag,hipporag2")
    print("  python scripts/serve.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
