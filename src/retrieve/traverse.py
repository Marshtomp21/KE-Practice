"""图遍历检索：锚点 -> 受限跳数邻域 -> 子图 + 子图证据所在的文本片段。

返回的文本片段不是重新做一次向量检索得来的，而是子图里每条边的证据片段，
这正是"图检索"与"向量检索"的差别所在：命中的文本由结构决定，而不是由词面
相似度决定。片段按其所支撑的边数排序，支撑得越多越靠前。
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.interfaces import Retriever
from ..core.types import Chunk, Entity, Relation, RetrievalResult
from .anchors import anchors_to_debug
from .registry import RetrievalContext, register


def chunks_supporting(
    relations: Sequence[Relation],
    entities: Sequence[Entity],
    lookup: Dict[str, Chunk],
    limit: int,
) -> Tuple[List[Chunk], Dict[str, float]]:
    """按证据出现次数给片段打分，取前 limit 段。"""
    weight: Counter[str] = Counter()
    for relation in relations:
        for evidence in relation.evidences:
            weight[evidence.chunk_id] += evidence.confidence
    for entity in entities:
        for evidence in entity.evidences:
            weight[evidence.chunk_id] += 0.2 * evidence.confidence

    ordered = [cid for cid, _ in weight.most_common() if cid in lookup][:limit]
    top = weight.most_common(1)[0][1] if weight else 1.0
    return (
        [lookup[cid] for cid in ordered],
        {cid: round(weight[cid] / top, 4) for cid in ordered},
    )


@register("traverse")
class GraphTraverseRetriever(Retriever):
    def __init__(self, context: RetrievalContext) -> None:
        self.context = context
        settings = context.settings
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))
        self.max_hops = int(settings.get("retrieval.traverse.max_hops", 2))
        self.max_nodes = int(settings.get("retrieval.traverse.max_nodes", 80))
        self.anchor_top_n = int(settings.get("retrieval.anchor_top_n", 5))
        self._lookup = context.chunk_lookup()

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        year_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> RetrievalResult:
        limit = top_k or self.default_top_k
        anchors = self.context.anchors.resolve(question, top_n=self.anchor_top_n)
        if not anchors:
            return RetrievalResult(
                retriever_name=self.name,
                debug_info={
                    "anchors": [],
                    "reason": "问句中没有识别到任何图内实体，无法展开邻域",
                    "max_hops": self.max_hops,
                },
            )

        seeds = [anchor.entity.id for anchor in anchors]
        entities, relations = self.context.store.neighborhood(
            seeds, hops=self.max_hops, max_nodes=self.max_nodes
        )
        chunks, chunk_scores = chunks_supporting(relations, entities, self._lookup, limit)

        # 节点分值按到锚点的跳距递减，前端据此决定节点大小
        distance = self._hop_distance(seeds, relations, {e.id for e in entities})
        node_scores = {
            entity.id: round(1.0 / (1.0 + distance.get(entity.id, self.max_hops)), 4)
            for entity in entities
        }

        return RetrievalResult(
            retriever_name=self.name,
            chunks=chunks,
            entities=entities,
            relations=relations,
            scores={**node_scores, **chunk_scores},
            debug_info={
                "anchors": anchors_to_debug(anchors),
                "max_hops": self.max_hops,
                "expanded_nodes": len(entities),
                "expanded_relations": len(relations),
                "seed_ids": seeds,
                "hop_distance": distance,
            },
        )

    def _hop_distance(
        self, seeds: Sequence[str], relations: Sequence[Relation], scope: set
    ) -> Dict[str, int]:
        adjacency: Dict[str, List[str]] = {}
        for relation in relations:
            adjacency.setdefault(relation.head_id, []).append(relation.tail_id)
            adjacency.setdefault(relation.tail_id, []).append(relation.head_id)

        distance = {seed: 0 for seed in seeds if seed in scope}
        frontier = list(distance)
        while frontier:
            nxt: List[str] = []
            for node in frontier:
                for neighbor in adjacency.get(node, []):
                    if neighbor in distance or neighbor not in scope:
                        continue
                    distance[neighbor] = distance[node] + 1
                    nxt.append(neighbor)
            frontier = nxt
        return distance
