"""KG²RAG-style complete QA method."""
from __future__ import annotations

from ..retrieve.kg2rag import KG2RAGRetriever
from ..retrieve.registry import RetrievalContext
from ._local_graph import LocalGraphQAMethod
from .registry import register


@register("kg2rag")
class KG2RAGMethod(LocalGraphQAMethod):
    """Semantic seeds, bounded graph expansion, reranking and generation."""

    def _build_retriever(self, context: RetrievalContext) -> KG2RAGRetriever:
        return KG2RAGRetriever(context)
