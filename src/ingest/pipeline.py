"""把 loader / cleaner / chunker 串成一条可复用的数据管线。

产物落盘为 jsonl，后续抽取阶段据此断点续跑，无需重跑清洗与切分。
"""
from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional, Tuple

from ..core.config import Settings, load_settings
from ..core.types import Chunk, CleanedDocument
from .chunker import SemanticChunker
from .cleaner import TextCleaner
from .loader import JsonDirectorySource, LoadReport


class IngestPipeline:
    """读入 -> 清洗 -> 切分。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        variant_path = (
            str(self.settings.path("paths.variant_table"))
            if self.settings.get("ingest.fold_variants", False)
            else None
        )
        self.cleaner = TextCleaner(
            self.settings.get("ingest.drop_patterns", []), variant_table_path=variant_path
        )
        self.chunker = SemanticChunker(
            target_chars=self.settings.get("ingest.target_chunk_chars", 400),
            max_chars=self.settings.get("ingest.max_chunk_chars", 700),
            min_chars=self.settings.get("ingest.min_chunk_chars", 60),
        )

    def run(
        self, already_done: Optional[List[str]] = None
    ) -> Tuple[List[CleanedDocument], List[Chunk], LoadReport]:
        report = LoadReport()
        source = JsonDirectorySource(
            self.settings.path("paths.raw_dir"),
            already_done=already_done,
            report=report,
        )
        documents: List[CleanedDocument] = []
        chunks: List[Chunk] = []
        for raw in source.iter_documents():
            try:
                cleaned = self.cleaner.clean(raw)
                produced = self.chunker.split(cleaned)
            except Exception as exc:  # 单条失败只记账，不中断整体导入
                report.accepted -= 1
                report.skip(raw.doc_id, f"清洗或切分失败: {exc}")
                continue
            documents.append(cleaned)
            chunks.extend(produced)
        return documents, chunks, report

    def persist(
        self, documents: List[CleanedDocument], chunks: List[Chunk]
    ) -> Dict[str, str]:
        chunk_file = self.settings.path("paths.chunk_file")
        doc_file = chunk_file.parent / "cleaned_docs.jsonl"
        chunk_file.parent.mkdir(parents=True, exist_ok=True)

        with chunk_file.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        with doc_file.open("w", encoding="utf-8") as handle:
            for doc in documents:
                handle.write(
                    json.dumps(
                        {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "url": doc.url,
                            "entity_type": doc.entity_type,
                            "text": doc.text,
                            "offset_map": doc.offset_map,
                            "infobox": doc.infobox,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return {"chunks": str(chunk_file), "documents": str(doc_file)}


def load_persisted_chunks(settings: Optional[Settings] = None) -> List[Chunk]:
    settings = settings or load_settings()
    target = settings.path("paths.chunk_file")
    if not target.exists():
        return []
    return [
        Chunk.from_dict(json.loads(line))
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iter_persisted_documents(
    settings: Optional[Settings] = None,
) -> Iterator[CleanedDocument]:
    settings = settings or load_settings()
    target = settings.path("paths.chunk_file").parent / "cleaned_docs.jsonl"
    if not target.exists():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        yield CleanedDocument(
            doc_id=payload["doc_id"],
            title=payload["title"],
            url=payload["url"],
            entity_type=payload["entity_type"],
            text=payload["text"],
            offset_map=payload["offset_map"],
            infobox=payload.get("infobox", {}),
        )
