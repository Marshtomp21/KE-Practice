"""只用于 benchmark 的简单混合与 oracle 上界，不注册到产品方法列表。"""
from __future__ import annotations

import time
from typing import Dict, Optional

from src.core.config import Settings
from src.core.interfaces import QAMethod
from src.core.types import Answer, Chunk, RetrievalConstraints, RetrievalResult
from src.methods.hipporag2 import HippoRAG2Method


class BenchmarkHybridMethod(QAMethod):
    """HippoRAG2 可见子图与文本检索结果做固定 RRF 合并。"""

    def __init__(self, settings: Settings, oracle: bool = False) -> None:
        self.oracle = oracle
        self.name = "oracle_repair" if oracle else "naive_hybrid"
        self.graph_method = HippoRAG2Method(settings)
        self.retriever = self.graph_method.retriever
        self.generator = self.graph_method.generator
        self.index = self.retriever.context.index
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        constraints: Optional[RetrievalConstraints] = None,
    ) -> Answer:
        started = time.perf_counter()
        limit = top_k or self.default_top_k
        constraints = constraints or RetrievalConstraints()
        graph_result = self.retriever.retrieve(question, top_k=limit, constraints=constraints)
        queries = list(constraints.supplemental_queries) if self.oracle else [question]

        rrf: Dict[str, float] = {}
        chunks: Dict[str, Chunk] = {chunk.id: chunk for chunk in graph_result.chunks}
        for rank, chunk in enumerate(graph_result.chunks, 1):
            rrf[chunk.id] = rrf.get(chunk.id, 0.0) + 1.0 / (60 + rank)
        query_documents: list[list[str]] = []
        supplemental_chunk_ids: set[str] = set()
        for query in queries:
            documents: list[str] = []
            for rank, (chunk, _) in enumerate(self.index.search(query, top_k=limit), 1):
                chunks[chunk.id] = chunk
                supplemental_chunk_ids.add(chunk.id)
                rrf[chunk.id] = rrf.get(chunk.id, 0.0) + 1.0 / (60 + rank)
                if chunk.doc_id not in documents:
                    documents.append(chunk.doc_id)
            query_documents.append(documents)
        ordered = sorted(rrf, key=lambda chunk_id: (-rrf[chunk_id], chunk_id))[:limit]
        selected_documents = list(dict.fromkeys(chunks[chunk_id].doc_id for chunk_id in ordered))
        compensation_documents = list(dict.fromkeys(
            chunks[chunk_id].doc_id for chunk_id in ordered if chunk_id in supplemental_chunk_ids
        ))
        temporary_relations = []
        if self.oracle and queries:
            temporary_relations = [
                {
                    "head_id": edge.head_id,
                    "relation": edge.relation,
                    "tail_id": edge.tail_id,
                    "supporting_documents": [
                        doc_id for doc_id in query_documents[index]
                        if doc_id in selected_documents
                    ],
                }
                for index, edge in enumerate(constraints.masked_edges)
            ]
        result = RetrievalResult(
            retriever_name=self.name,
            chunks=[chunks[chunk_id] for chunk_id in ordered],
            entities=graph_result.entities,
            relations=graph_result.relations,
            scores={**graph_result.scores, **{key: rrf[key] for key in ordered}},
            debug_info={
                "method": self.name,
                "fusion": "rrf",
                "gap_detected": bool(queries) if self.oracle else False,
                "compensation_triggered": bool(queries),
                "compensation_queries": queries,
                "compensation_documents": compensation_documents,
                "temporary_relations": temporary_relations,
                "graph_debug": graph_result.debug_info,
            },
        )
        answer = self.generator.generate(question, result)
        answer.retriever_name = self.name
        answer.latency = time.perf_counter() - started
        answer.debug_info.setdefault("retrieval", result.debug_info)
        return answer

    def close(self) -> None:
        self.graph_method.close()
