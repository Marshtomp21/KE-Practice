"""混合检索：向量结果与图结果的融合。

两种融合策略，由 settings.yaml 的 retrieval.hybrid.strategy 选择：
- rrf：倒数排名融合。只看名次不看分值，因此不需要在余弦相似度与 PPR 概率
  之间做量纲对齐，是默认策略。
- weighted：先把两路分值各自归一到 [0,1] 再加权求和，权重可配。

子图部分直接沿用图路的结果——向量路本来就不产生子图，不存在"融合"的问题。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..core.interfaces import Retriever
from ..core.types import Chunk, RetrievalResult
from .registry import RetrievalContext, build_retriever, register


def _rank_map(chunk_ids: Sequence[str]) -> Dict[str, int]:
    return {chunk_id: rank for rank, chunk_id in enumerate(chunk_ids, start=1)}


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    peak = max(scores.values())
    floor = min(scores.values())
    span = peak - floor
    if span <= 0:
        return {key: 1.0 for key in scores}
    return {key: (value - floor) / span for key, value in scores.items()}


@register("hybrid")
class HybridRetriever(Retriever):
    def __init__(self, context: RetrievalContext) -> None:
        self.context = context
        settings = context.settings
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))
        self.strategy = str(settings.get("retrieval.hybrid.strategy", "rrf")).lower()
        self.rrf_k = int(settings.get("retrieval.hybrid.rrf_k", 60))
        self.vector_weight = float(settings.get("retrieval.hybrid.vector_weight", 0.5))
        self.graph_weight = float(settings.get("retrieval.hybrid.graph_weight", 0.5))
        # 复用已注册的两路检索器，融合层不重复实现检索逻辑
        self.vector_arm = build_retriever("vector", context)
        self.graph_arm = build_retriever("ppr", context)

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        year_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> RetrievalResult:
        limit = top_k or self.default_top_k
        # 两路各自多取一些候选，融合后再截断，避免融合前就被截掉好结果
        vector_result = self.vector_arm.retrieve(question, top_k=limit * 2, year_range=year_range)
        graph_result = self.graph_arm.retrieve(question, top_k=limit * 2, year_range=year_range)

        pool: Dict[str, Chunk] = {c.id: c for c in vector_result.chunks}
        pool.update({c.id: c for c in graph_result.chunks})

        vector_ids = [c.id for c in vector_result.chunks]
        graph_ids = [c.id for c in graph_result.chunks]
        fused = self._fuse(vector_ids, graph_ids, vector_result.scores, graph_result.scores)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
        chunks = [pool[cid] for cid, _ in ordered if cid in pool]
        chunk_scores = {cid: round(score, 4) for cid, score in ordered if cid in pool}
        node_scores = {
            key: value
            for key, value in graph_result.scores.items()
            if key not in pool
        }

        return RetrievalResult(
            retriever_name=self.name,
            chunks=chunks,
            entities=graph_result.entities,
            relations=graph_result.relations,
            scores={**node_scores, **chunk_scores},
            debug_info={
                "strategy": self.strategy,
                "vector_arm": vector_result.retriever_name,
                "graph_arm": graph_result.retriever_name,
                "vector_hits": vector_ids,
                "graph_hits": graph_ids,
                "overlap": sorted(set(vector_ids) & set(graph_ids)),
                "graph_debug": graph_result.debug_info,
            },
        )

    def _fuse(
        self,
        vector_ids: Sequence[str],
        graph_ids: Sequence[str],
        vector_scores: Dict[str, float],
        graph_scores: Dict[str, float],
    ) -> Dict[str, float]:
        if self.strategy == "weighted":
            left = _normalize({cid: vector_scores.get(cid, 0.0) for cid in vector_ids})
            right = _normalize({cid: graph_scores.get(cid, 0.0) for cid in graph_ids})
            keys = set(left) | set(right)
            return {
                key: self.vector_weight * left.get(key, 0.0)
                + self.graph_weight * right.get(key, 0.0)
                for key in keys
            }

        left_rank = _rank_map(vector_ids)
        right_rank = _rank_map(graph_ids)
        keys = set(left_rank) | set(right_rank)
        fused: Dict[str, float] = {}
        for key in keys:
            score = 0.0
            if key in left_rank:
                score += self.vector_weight / (self.rrf_k + left_rank[key])
            if key in right_rank:
                score += self.graph_weight / (self.rrf_k + right_rank[key])
            fused[key] = score
        return fused
