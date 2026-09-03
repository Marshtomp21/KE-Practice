"""个性化 PageRank 检索。

以问句锚点为重启分布在图上跑 PPR，得分高的节点构成子图。与固定跳数的邻域
扩展相比，它对"经由哪些中间人产生关联"这类问题更合适：得分自然把高连通的
桥接节点顶上来，而不是把锚点周围所有邻居一视同仁地铺开。

得分原样写进 RetrievalResult.scores，前端直接用它决定节点大小。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..core.interfaces import Retriever
from ..core.types import RetrievalResult
from .anchors import anchors_to_debug
from .registry import RetrievalContext, register
from .traverse import chunks_supporting


@register("ppr")
class PPRRetriever(Retriever):
    def __init__(self, context: RetrievalContext) -> None:
        self.context = context
        settings = context.settings
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))
        self.alpha = float(settings.get("retrieval.ppr.alpha", 0.85))
        self.max_iter = int(settings.get("retrieval.ppr.max_iter", 60))
        self.tolerance = float(settings.get("retrieval.ppr.tolerance", 1e-6))
        self.top_nodes = int(settings.get("retrieval.ppr.top_nodes", 40))
        self.anchor_top_n = int(settings.get("retrieval.anchor_top_n", 5))
        self._lookup = context.chunk_lookup()
        # PPR 在无向视图上跑：影视关系的"关联性"没有方向，
        # 沿有向边走会让"某片的演员"无法反向到达该片的导演。
        self._undirected = self._to_weighted_undirected(context.store.as_networkx())

    def _to_weighted_undirected(self, graph: nx.MultiDiGraph) -> nx.Graph:
        collapsed = nx.Graph()
        collapsed.add_nodes_from(graph.nodes())
        for head, tail, data in graph.edges(data=True):
            relation = data.get("payload")
            weight = max((e.confidence for e in relation.evidences), default=0.5) if relation else 0.5
            if collapsed.has_edge(head, tail):
                collapsed[head][tail]["weight"] += weight
            else:
                collapsed.add_edge(head, tail, weight=weight)
        return collapsed

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        year_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> RetrievalResult:
        limit = top_k or self.default_top_k
        anchors = self.context.anchors.resolve(question, top_n=self.anchor_top_n)
        if not anchors or self._undirected.number_of_nodes() == 0:
            return RetrievalResult(
                retriever_name=self.name,
                debug_info={
                    "anchors": [],
                    "reason": "没有可用作重启分布的锚点实体",
                },
            )

        personalization: Dict[str, float] = {}
        for anchor in anchors:
            if anchor.entity.id in self._undirected:
                personalization[anchor.entity.id] = personalization.get(anchor.entity.id, 0.0) + anchor.score
        if not personalization:
            return RetrievalResult(
                retriever_name=self.name,
                debug_info={"anchors": anchors_to_debug(anchors), "reason": "锚点不在图中"},
            )

        converged = True
        try:
            ranks = nx.pagerank(
                self._undirected,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tolerance,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence:
            converged = False
            ranks = nx.pagerank(
                self._undirected,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter * 5,
                tol=self.tolerance * 100,
                weight="weight",
            )

        ordered = sorted(ranks.items(), key=lambda kv: -kv[1])[: self.top_nodes]
        keep = [node for node, score in ordered if score > 0]
        entities = [e for e in (self.context.store.get_entity(n) for n in keep) if e]
        relations = self.context.store.subgraph_relations(keep)
        chunks, chunk_scores = chunks_supporting(relations, entities, self._lookup, limit)

        peak = ordered[0][1] if ordered else 1.0
        node_scores = {node: round(ranks[node] / peak, 4) for node in keep}

        return RetrievalResult(
            retriever_name=self.name,
            chunks=chunks,
            entities=entities,
            relations=relations,
            scores={**node_scores, **chunk_scores},
            debug_info={
                "anchors": anchors_to_debug(anchors),
                "alpha": self.alpha,
                "converged": converged,
                "max_iter": self.max_iter,
                "ranked_nodes": len(ranks),
                "kept_nodes": len(keep),
                "top_nodes": [
                    {"id": node, "score": round(score, 6)} for node, score in ordered[:10]
                ],
            },
        )
