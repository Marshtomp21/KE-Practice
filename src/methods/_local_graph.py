"""Shared lifecycle for local graph research methods."""
from __future__ import annotations

import time
from typing import Optional

from ..core.config import Settings
from ..core.interfaces import AnswerGenerator, QAMethod, Retriever
from ..core.types import Answer
from ..generate.answer import build_generator
from ..retrieve.anchors import AnchorResolver
from ..retrieve.dataset_graph import DatasetGraphLoader
from ..retrieve.registry import RetrievalContext
from ..retrieve.vector_index import ChunkVectorIndex


class LocalGraphQAMethod(QAMethod):
    """Load isolated local resources, retrieve, then use the shared generator."""

    def __init__(
        self,
        settings: Settings,
        retriever: Optional[Retriever] = None,
        generator: Optional[AnswerGenerator] = None,
    ) -> None:
        self.settings = settings
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))
        if retriever is None:
            index = ChunkVectorIndex(settings=settings).load()
            store = DatasetGraphLoader(settings, index).load()
            context = RetrievalContext(
                store=store,
                index=index,
                settings=settings,
                anchors=AnchorResolver(
                    store,
                    min_score=float(settings.get("retrieval.anchor_min_score", 0.55)),
                ),
            )
            retriever = self._build_retriever(context)
        self.retriever = retriever
        self.generator = generator or build_generator(settings)

    def _build_retriever(self, context: RetrievalContext) -> Retriever:
        raise NotImplementedError

    def ask(self, question: str, top_k: Optional[int] = None) -> Answer:
        started = time.perf_counter()
        limit = top_k or self.default_top_k
        result = self.retriever.retrieve(question, top_k=limit)
        answer = self.generator.generate(question, result)
        answer.retriever_name = self.name
        answer.latency = time.perf_counter() - started
        answer.debug_info.setdefault("retrieval", result.debug_info)
        return answer
