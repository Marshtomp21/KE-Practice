"""打印数据处理前后的对比样例：原文 / 清洗后 / 切分结果 / 偏移反查。

用于成果展示，也用于人工确认偏移映射没有错位。
用法：python scripts/inspect_ingest.py [--limit 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_settings
from src.ingest.chunker import SemanticChunker
from src.ingest.cleaner import TextCleaner
from src.ingest.loader import JsonDirectorySource, LoadReport


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2, help="展示前几个条目")
    parser.add_argument("--preview", type=int, default=260, help="每段最多展示的字符数")
    args = parser.parse_args()

    settings = load_settings()
    cleaner = TextCleaner(
        settings.get("ingest.drop_patterns", []),
        variant_table_path=(
            str(settings.path("paths.variant_table"))
            if settings.get("ingest.fold_variants", False)
            else None
        ),
    )
    chunker = SemanticChunker(
        target_chars=settings.get("ingest.target_chunk_chars", 400),
        max_chars=settings.get("ingest.max_chunk_chars", 700),
        min_chars=settings.get("ingest.min_chunk_chars", 60),
    )
    report = LoadReport()
    source = JsonDirectorySource(settings.path("paths.raw_dir"), report=report)

    shown = 0
    for raw in source.iter_documents():
        if shown >= args.limit:
            break
        shown += 1
        cleaned = cleaner.clean(raw)
        chunks = chunker.split(cleaned)

        rule(f"[{shown}] {raw.doc_id}  {raw.title}  ({raw.entity_type})")
        print("--- 原文 ---")
        print(repr(raw.text[: args.preview]))
        print(f"\n--- 清洗后（{len(raw.text)} -> {len(cleaned.text)} 字符）---")
        print(repr(cleaned.text[: args.preview]))

        print(f"\n--- 切分为 {len(chunks)} 个片段 ---")
        for chunk in chunks:
            lo, hi = chunk.metadata["raw_span"]
            print(
                f"  {chunk.id}  长度 {len(chunk.text):>3}  "
                f"清洗后 [{chunk.char_offset}:{chunk.char_end}]  原文 [{lo}:{hi}]"
            )
            print(f"    {chunk.text[:80]}")

        if chunks:
            probe = chunks[0]
            lo, hi = cleaned.to_raw_span(probe.char_offset, probe.char_offset + 12)
            print("\n--- 偏移反查校验（取首片段前 12 字）---")
            print(f"  清洗后: {cleaned.text[probe.char_offset:probe.char_offset + 12]!r}")
            print(f"  原文  : {raw.text[lo:hi]!r}")

    rule("导入账本")
    print(report.summary())
    for item in report.skipped[:10]:
        print(f"  跳过 {item['source']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
