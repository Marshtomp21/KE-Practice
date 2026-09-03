"""本地向量 RAG 基线。"""
from __future__ import annotations

import time
from typing import Optional

from ..core.config import Settings
from ..core.interfaces import AnswerGenerator, QAMethod
from ..core.types import Answer, RetrievalResult
from ..generate.answer import build_generator
from ..retrieve.vector_index import ChunkVectorIndex
from .registry import register


@register("vector")
class VectorQAMethod(QAMethod):
    """用项目内的向量索引召回文本，再交给统一答案生成器。"""

    def __init__(
        self,
        settings: Settings,
        index: Optional[ChunkVectorIndex] = None,
        generator: Optional[AnswerGenerator] = None,
    ) -> None:
        self.settings = settings
        self.index = index or ChunkVectorIndex(settings=settings).load()
        self.generator = generator or build_generator(settings)
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))

    def ask(self, question: str, top_k: Optional[int] = None) -> Answer:
        started = time.perf_counter()
        limit = top_k or self.default_top_k
        hits = self.index.search(question, top_k=limit)
        result = RetrievalResult(
            retriever_name=self.name,
            chunks=[chunk for chunk, _ in hits],
            scores={chunk.id: score for chunk, score in hits},
            debug_info={
                "method": self.name,
                "top_k": limit,
                "hit_count": len(hits),
                "backend": "local-numpy",
            },
        )
        answer = self.generator.generate(question, result)
        answer.retriever_name = self.name
        answer.latency = time.perf_counter() - started
        answer.debug_info.setdefault("retrieval", result.debug_info)
        return answer
