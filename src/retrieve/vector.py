"""对照组：纯向量检索，完全不看图。

它是对比实验的基线，因此实现上刻意不做任何额外照顾，也不做任何削弱：
用与其它检索器同一份切分、同一个 embedder、同一个 top_k，只是不查图而已。
"""
from __future__ import annotations

from typing import Optional, Tuple

from ..core.interfaces import Retriever
from ..core.types import RetrievalResult
from .registry import RetrievalContext, register


@register("vector")
class VectorRetriever(Retriever):
    def __init__(self, context: RetrievalContext) -> None:
        self.context = context
        self.default_top_k = int(context.settings.get("retrieval.top_k_chunks", 6))

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        year_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> RetrievalResult:
        limit = top_k or self.default_top_k
        hits = self.context.index.search(question, top_k=limit)
        return RetrievalResult(
            retriever_name=self.name,
            chunks=[chunk for chunk, _ in hits],
            entities=[],
            relations=[],
            scores={chunk.id: score for chunk, score in hits},
            debug_info={
                "top_k": limit,
                "hit_count": len(hits),
                "uses_graph": False,
                "note": "基线检索，不查图，因此不返回实体与关系",
            },
        )
