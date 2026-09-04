"""HippoRAG 2-style complete QA method."""
from __future__ import annotations

from ..retrieve.hipporag2 import HippoRAG2Retriever
from ..retrieve.registry import RetrievalContext
from ._local_graph import LocalGraphQAMethod
from .registry import register


@register("hipporag2")
class HippoRAG2Method(LocalGraphQAMethod):
    """Entity-seeded PPR retrieval followed by the shared generator."""

    def _build_retriever(self, context: RetrievalContext) -> HippoRAG2Retriever:
        return HippoRAG2Retriever(context)
