"""HippoRAG 2-style personalized graph propagation retrieval."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.interfaces import Retriever
from ..core.types import Entity, Relation, RetrievalConstraints, RetrievalResult
from .anchors import anchors_to_debug
from .registry import RetrievalContext
from .research_utils import evidence_chunk_ids, normalize_scores, relation_adjacency, semantic_scores


class HippoRAG2Retriever(Retriever):
    """Entity-seeded PPR with hub suppression and auditable bridge paths."""

    name = "hipporag2"

    def __init__(self, context: RetrievalContext) -> None:
        self.context = context
        settings = context.settings
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))
        self.anchor_top_n = int(settings.get("hipporag2.anchor_top_n", 5))
        self.alpha = float(settings.get("hipporag2.alpha", 0.85))
        self.max_iter = int(settings.get("hipporag2.max_iter", 80))
        self.tolerance = float(settings.get("hipporag2.tolerance", 1e-8))
        self.top_nodes = int(settings.get("hipporag2.top_nodes", 50))
        self.degree_penalty = float(settings.get("hipporag2.degree_penalty", 0.5))
        self.max_path_hops = int(settings.get("hipporag2.max_path_hops", 4))
        self.graph_weight = float(settings.get("hipporag2.graph_weight", 0.85))
        self.semantic_weight = float(settings.get("hipporag2.semantic_weight", 0.15))
        self._chunks = context.chunk_lookup()
        self._entities = {entity.id: entity for entity in context.store.all_entities()}
        self._relations = context.store.all_relations()

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        year_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        constraints: Optional[RetrievalConstraints] = None,
    ) -> RetrievalResult:
        limit = top_k or self.default_top_k
        constraints = constraints or RetrievalConstraints()
        active_relations = [
            relation for relation in self._relations
            if not constraints.masks(relation.head_id, relation.type, relation.tail_id)
        ]
        graph_adjacency = relation_adjacency(active_relations)
        walk = self._weighted_adjacency(active_relations)
        anchors = self.context.anchors.resolve(question, top_n=self.anchor_top_n)
        personalization = {
            anchor.entity.id: anchor.score
            for anchor in anchors if anchor.entity.id in walk
        }
        fallback_chunks = []
        if not personalization:
            fallback_chunks = self.context.index.search(question, top_k=max(3, limit))
            chunk_ids = {chunk.id for chunk, _ in fallback_chunks}
            for entity in self._entities.values():
                if any(evidence.chunk_id in chunk_ids for evidence in entity.evidences):
                    personalization[entity.id] = 0.5

        if not personalization:
            return RetrievalResult(
                retriever_name=self.name,
                chunks=[chunk for chunk, _ in fallback_chunks[:limit]],
                scores={chunk.id: float(score) for chunk, score in fallback_chunks[:limit]},
                debug_info={
                    "method": "entity_seeded_ppr",
                    "anchors": anchors_to_debug(anchors),
                    "graph_seeds": [],
                    "reason": "没有可映射到图节点的查询实体，退回语义证据",
                },
            )

        ranks, iterations, converged = self._pagerank(personalization, walk)
        ranked = sorted(ranks.items(), key=lambda item: (-item[1], item[0]))
        keep = {node for node, score in ranked[: self.top_nodes] if score > 0}
        bridge_paths = self._bridge_paths(list(personalization), graph_adjacency)
        for path in bridge_paths:
            keep.update(path)

        entities = [self._entities[node] for node in keep if node in self._entities]
        relations = [
            relation for relation in active_relations
            if relation.head_id in keep and relation.tail_id in keep
        ]
        candidate_ids = evidence_chunk_ids(relations, entities)
        graph_scores = self._chunk_scores(relations, entities, ranks)
        semantic = semantic_scores(self.context.index, question, candidate_ids)
        graph_norm = normalize_scores(graph_scores)
        semantic_norm = normalize_scores(semantic)
        fused = {
            chunk_id: self.graph_weight * graph_norm.get(chunk_id, 0.0)
            + self.semantic_weight * semantic_norm.get(chunk_id, 0.0)
            for chunk_id in candidate_ids
        }
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit]
        chunks = [self._chunks[chunk_id] for chunk_id, _ in ordered if chunk_id in self._chunks]
        peak = ranked[0][1] if ranked else 1.0
        node_scores = {node: ranks[node] / peak for node in keep if peak > 0}

        return RetrievalResult(
            retriever_name=self.name,
            chunks=chunks,
            entities=entities,
            relations=relations,
            scores={
                **{key: round(value, 6) for key, value in node_scores.items()},
                **{key: round(value, 6) for key, value in ordered},
            },
            debug_info={
                "method": "entity_seeded_ppr_with_hub_penalty",
                "anchors": anchors_to_debug(anchors),
                "graph_seeds": [
                    {"entity_id": node, "weight": round(weight, 6)}
                    for node, weight in sorted(personalization.items())
                ],
                "alpha": self.alpha,
                "degree_penalty": self.degree_penalty,
                "masked_edge_count": len(constraints.masked_edges),
                "iterations": iterations,
                "converged": converged,
                "ranked_nodes": len(ranks),
                "kept_nodes": len(keep),
                "top_nodes": [
                    {"entity_id": node, "score": round(score, 8)}
                    for node, score in ranked[:10]
                ],
                "bridge_paths": bridge_paths,
                "reranked_chunks": [
                    {
                        "chunk_id": chunk_id,
                        "score": round(score, 6),
                        "graph": round(graph_norm.get(chunk_id, 0.0), 6),
                        "semantic": round(semantic_norm.get(chunk_id, 0.0), 6),
                    }
                    for chunk_id, score in ordered
                ],
            },
        )

    def _weighted_adjacency(self, relations: Sequence[Relation]) -> Dict[str, Dict[str, float]]:
        neighbors: Dict[str, set[str]] = defaultdict(set)
        raw: Dict[Tuple[str, str], float] = defaultdict(float)
        for relation in relations:
            left, right = relation.head_id, relation.tail_id
            neighbors[left].add(right)
            neighbors[right].add(left)
            confidence = max((item.confidence for item in relation.evidences), default=0.5)
            raw[tuple(sorted((left, right)))] += confidence

        weighted: Dict[str, Dict[str, float]] = {node: {} for node in self._entities}
        for (left, right), value in raw.items():
            penalty = (max(len(neighbors[left]), 1) * max(len(neighbors[right]), 1)) ** self.degree_penalty
            weight = value / penalty
            weighted[left][right] = weighted[left].get(right, 0.0) + weight
            weighted[right][left] = weighted[right].get(left, 0.0) + weight
        return weighted

    def _pagerank(
        self, personalization: Dict[str, float], walk: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, float], int, bool]:
        total = sum(personalization.values()) or 1.0
        restart = {node: value / total for node, value in personalization.items()}
        ranks = {node: restart.get(node, 0.0) for node in walk}
        nodes = list(walk)
        if not nodes:
            return {}, 0, True

        for iteration in range(1, self.max_iter + 1):
            updated = {node: (1.0 - self.alpha) * restart.get(node, 0.0) for node in nodes}
            dangling = 0.0
            for node, score in ranks.items():
                outgoing = walk.get(node, {})
                weight_sum = sum(outgoing.values())
                if weight_sum <= 0:
                    dangling += score
                    continue
                for neighbor, weight in outgoing.items():
                    updated[neighbor] += self.alpha * score * weight / weight_sum
            if dangling:
                for node, weight in restart.items():
                    updated[node] += self.alpha * dangling * weight
            delta = sum(abs(updated[node] - ranks.get(node, 0.0)) for node in nodes)
            ranks = updated
            if delta <= self.tolerance:
                return ranks, iteration, True
        return ranks, self.max_iter, False

    def _bridge_paths(
        self, seeds: Sequence[str], adjacency: Dict[str, list]
    ) -> List[List[str]]:
        paths: List[List[str]] = []
        for index, source in enumerate(seeds):
            for target in seeds[index + 1 :]:
                path = self._shortest_path(source, target, adjacency)
                if path:
                    paths.append(path)
        return paths

    def _shortest_path(
        self, source: str, target: str, adjacency: Dict[str, list]
    ) -> List[str]:
        queue = deque([(source, [source])])
        seen = {source}
        while queue:
            node, path = queue.popleft()
            if len(path) - 1 >= self.max_path_hops:
                continue
            for neighbor, _ in sorted(adjacency.get(node, []), key=lambda pair: pair[0]):
                if neighbor == target:
                    return [*path, neighbor]
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, [*path, neighbor]))
        return []

    @staticmethod
    def _chunk_scores(
        relations: Sequence[Relation], entities: Sequence[Entity], ranks: Dict[str, float]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for relation in relations:
            support = ranks.get(relation.head_id, 0.0) + ranks.get(relation.tail_id, 0.0)
            for evidence in relation.evidences:
                scores[evidence.chunk_id] += support * evidence.confidence
        for entity in entities:
            for evidence in entity.evidences:
                scores[evidence.chunk_id] += 0.2 * ranks.get(entity.id, 0.0)
        return dict(scores)
